import argparse
import logging
import os
import sys
import numpy as np

from omegaconf import OmegaConf

import mindspore as ms
from mindspore import ops

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(workspace))

sd_abs_path = os.path.dirname(os.path.dirname(workspace)) + "/stable_diffusion_v2"
sys.path.append(sd_abs_path)

from libs.helper import set_env
from libs.infer_engine.export_modules import (
    MotionStyleTransferDataPrepare,
    NoisySample,
    MotionStyleTransferPredictNoise,
    SchedulerPreProcess,
    VAEDecoder,
)

from vc.models import AutoencoderKL, FrozenOpenCLIPEmbedder, FrozenOpenCLIPVisualEmbedder, UNetSD_temporal
from vc.trainer.lr_scheduler import build_lr_scheduler
from vc.trainer.optim import build_optimizer
from vc.utils import get_abspath_of_weights, setup_logger

from ldm.modules.logger import set_logger
from ldm.util import instantiate_from_config, str2bool
from ldm.modules.train.tools import set_random_seed


logger = logging.getLogger("Video Composer Export")


def model_export(net, inputs, name, model_save_path):
    ms.export(net, *inputs, file_name=os.path.join(model_save_path, name), file_format="MINDIR")
    logger.info(f"convert {name} mindir done")


def lite_convert(name, model_save_path, converter):
    import mindspore_lite as mslite

    mindir_path = os.path.join(model_save_path, f"{name}.mindir")
    if not os.path.exists(mindir_path):
        mindir_path = os.path.join(model_save_path, f"{name}_graph.mindir")
    converter.convert(
        fmk_type=mslite.FmkType.MINDIR,
        model_file=mindir_path,
        output_file=os.path.join(model_save_path, f"{name}_lite"),
        config_file="./libs/infer_engine/sd_lite.cfg",
    )
    logger.info(f"convert {name} lite mindir done")


def create_models(cfg, task='motion_style_transfer'):
    # 2.1 clip - text encoder, and image encoder (optional)
    clip_text_encoder = FrozenOpenCLIPEmbedder(
        layer="penultimate",
        pretrained_ckpt_path=cfg.clip_checkpoint,
        tokenizer_path=cfg.clip_tokenizer,
        use_fp16=cfg.use_fp16,
    )
    logger.info("clip text encoder init.")
    tokenizer = clip_text_encoder.tokenizer
    clip_text_encoder.set_train(False)
    
    if args.task in ['motion_style_transfer']:
        clip_image_encoder = FrozenOpenCLIPVisualEmbedder(
            layer="penultimate", pretrained_ckpt_path=cfg.clip_checkpoint, use_fp16=cfg.use_fp16
        )
        clip_image_encoder.set_train(False)
        logger.info("clip image encoder init.")
    else:
        clip_image_encoder = None

    # 2.2 vae
    vae = AutoencoderKL(
        cfg.sd_config,
        4,
        ckpt_path=cfg.sd_checkpoint,
        use_fp16=cfg.use_fp16,
        version="2.1",
    )
    vae = vae.set_train(False)
    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False
    logger.info("vae init")

    # 2.3 unet3d with STC encoders
    assert cfg.use_fps_condition==False, f"use_fps_condtion==True require rebuild the graphs"
    unet = UNetSD_temporal(
        cfg=cfg,
        in_dim=cfg.unet_in_dim,
        concat_dim=cfg.unet_concat_dim,
        dim=cfg.unet_dim,
        context_dim=cfg.unet_context_dim,
        out_dim=cfg.unet_out_dim,
        dim_mult=cfg.unet_dim_mult,
        num_heads=cfg.unet_num_heads,
        head_dim=cfg.unet_head_dim,
        num_res_blocks=cfg.unet_res_blocks,
        attn_scales=cfg.unet_attn_scales,
        dropout=cfg.unet_dropout,
        temporal_attention=cfg.temporal_attention,
        temporal_attn_times=cfg.temporal_attn_times,
        use_fps_condition=cfg.use_fps_condition,
        use_sim_mask=cfg.use_sim_mask,
        video_compositions=cfg.video_compositions,
        misc_dropout=cfg.misc_dropout,
        p_all_zero=cfg.p_all_zero,
        p_all_keep=cfg.p_all_zero,
        zero_y=None,  # assume we always use text prompts  (y, even y="")
        use_fp16=cfg.use_fp16,
        use_adaptive_pool=cfg.use_adaptive_pool,
    )
    # TODO: use common checkpoiont download, mapping, and loading
    unet.load_state_dict(cfg.resume_checkpoint)
    unet = unet.set_train(False)
    for param in unet.get_parameters():  # freeze unet
        param.requires_grad = False

    # 2.4 other NN-based condition extractors
    extra_conds = {}

    return unet, vae, clip_text_encoder, clip_image_encoder, extra_conds

def main(args):
    # set logger
    set_env(args)
    set_random_seed(args.seed)
    ms.set_context(device_target="CPU")
    
    model_save_path = os.path.join(args.output_path, args.task)
    os.makedirs(model_save_path, exist_ok=True)
    logger.info(f"model_save_path: {model_save_path}")
       
    # get converter
    converter = None
    # lite convert setting 
    if args.converte_lite:
        import mindspore_lite as mslite

        optimize_dict = {"ascend": "ascend_oriented", "gpu": "gpu_oriented", "cpu": "general"}
        converter = mslite.Converter()
        converter.save_type = mslite.ModelType.MINDIR
        converter.optimize = optimize_dict[args.device_target.lower()]

    # VC model config file
    from config.model_config import cfg  # base arch config
    bs = batch_size = args.n_samples
     
    # 1. Create input data as place holder
    # 1) inputs for DataPrepare graph 
    # TODO: read image sizes from cfg or args
    tokenized_prompts = ops.ones((batch_size, 77), ms.int32)
    style_image = ops.ones((bs, 3, 224, 224), ms.float32)  
    single_image = ops.ones((bs, 1, 3, 384, 384), ms.float32)  
    motion_vectors = ops.ones((bs, cfg.max_frames, 2, 256, 256), ms.float32)  
    #fps = ops.ones((), ms.int32) # passed to noise predictor
    noise = ops.ones((batch_size, 4, cfg.max_frames, 256 // 8, 256 // 8), ms.float32) # passed to noise predictor

    # 2) inputs for noise prediction graph (unet) and scheduler
    text_emb = ops.ones((batch_size * 2, 77, 1024), ms.float16) # TODO: due to concat with img emb (to float16)
    style_emb = ops.ones((batch_size, 1, 1024), ms.float16) # TODO: due to clip vit output fp16
    single_image_tr = ops.ones((bs, 3, cfg.max_frames, 384, 384), ms.float32)  
    motion_vectors_tr = ops.ones((bs, 2, cfg.max_frames, 256, 256), ms.float32)  
    scale = ops.ones((), ms.float32) # unconditional guidance scale
    ts = ops.ones((), ms.int32)

    # 2. Create model components loaded with pretrained weight   
    unet, vae, text_encoder, clip_image_encoder, extra_conds = create_models(cfg, task=args.task)

    # create scheduler
    sampler_config = OmegaConf.load(args.sampler)
    scheduler = instantiate_from_config(sampler_config)
    scheduler_type = sampler_config.type

    export_mindir = True

    # 3. Build pipeline sub-graphs/Cell used to form the whole inference pipeline, from data prepare, noise prediction, to vae decoding 
    data_prepare, scheduler_preprocess, predict_noise, noisy_sample, vae_decoder = None, None, None, None, None 
    # 3.1 data prepare graph including text, image, noise, and nn-based condtion extractions
    if not data_prepare:
        if args.task == "motion_style_transfer":
            # TODO: use guidances to conrol the conditions extraction graph building
            data_prepare = MotionStyleTransferDataPrepare(text_encoder, vae, scheduler, cfg.scale_factor, clip_image_encoder, extra_conds=extra_conds, frames=cfg.max_frames)
            model_export(
                net=data_prepare,
                inputs=(tokenized_prompts, tokenized_prompts, noise, style_image, single_image, motion_vectors),
                name=args.inputs_config.data_prepare_model,
                model_save_path=model_save_path,
            )
        else:
            raise ValueError(f"Not support task: {args.task}")

    # 3.2 latent noise scaling  graph
    # TODO: why not put it into DataPrepare graph?
    # TODO: need scaling for VC??
    if not scheduler_preprocess:
        scheduler_preprocess = SchedulerPreProcess(scheduler)
        
        model_export(
            net=scheduler_preprocess,
            inputs=(noise, ts),
            name=f"{args.inputs_config.scheduler_preprocess}-{scheduler_type}",
            model_save_path=model_save_path,
            )

    # 3.3 noise prediciton graph (UNet forward), eps
    # TODO: extend for other tasks
    DEBUG = False
    if not predict_noise:
        predict_noise = MotionStyleTransferPredictNoise(unet)
        if DEBUG:
            eps = predict_noise(noise, ts, text_emb, style_emb, single_image_tr, motion_vectors_tr, scale)
            print("D--: ", eps)
        model_export(
            net=predict_noise,
            inputs=(noise, ts, text_emb, style_emb, single_image_tr, motion_vectors_tr, scale),
            name=args.inputs_config.predict_noise_model,
            model_save_path=model_save_path,
        )
    
    # 3.4 diffusion backward (sampling) graph, z_{t-j} = sample(z_t, eps, ts)
    if not noisy_sample:
        noisy_sample = NoisySample(scheduler)
        model_export(
            net=noisy_sample,
            inputs=(noise, ts, noise, ts),
            name=f"{args.inputs_config.noisy_sample_model}-{scheduler_type}",
            model_save_path=model_save_path,
        )

    # 3.5 latent to pixel decoding graph 
    if not vae_decoder:
        vae_decoder = VAEDecoder(vae, cfg.scale_factor)
        model_export(
            net=vae_decoder, inputs=(noise,), name=args.inputs_config.vae_decoder_model, model_save_path=model_save_path
        )

    if args.converte_lite:
        lite_convert(args.inputs_config.data_prepare_model, model_save_path, converter)
        lite_convert(f"{args.inputs_config.scheduler_preprocess}-{scheduler_type}", model_save_path, converter)
        lite_convert(args.inputs_config.predict_noise_model, model_save_path, converter)
        lite_convert(f"{args.inputs_config.noisy_sample_model}-{scheduler_type}", model_save_path, converter)
        lite_convert(args.inputs_config.vae_decoder_model, model_save_path, converter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument(
        "--device_target",
        type=str,
        default="Ascend",
        help="Device target, should be in [Ascend]",
        choices=["Ascend"],
    )
    parser.add_argument(
        "--task",
        type=str,
        default="motion_style_transfer",
        help="Task name, options: motion_transfer, motion_style_transfer"
        "if choose a task name, use the config/[task].yaml for inputs",
        choices=["motion_transfer", "motion_style_transfer"],
    )
    parser.add_argument("--model", type=str, required=False, help="path to config which constructs model.")
    #parser.add_argument("--only_converte_lite", default=False, type=str2bool, help="whether convert MindSpore mindir")
    parser.add_argument("--converte_lite", default=True, type=str2bool, help="whether convert lite mindir")
    parser.add_argument("--output_path", type=str, default="output", help="dir to write results to")
    parser.add_argument("--sampler", type=str, default="config/schedule/ddim.yaml", help="infer sampler yaml path")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt in an iteration. A.k.a. batch size",
    )
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")
    parser.add_argument("--log_level", type=str, default="INFO", help="log level, options: DEBUG, INFO, WARNING, ERROR")
    args = parser.parse_args()
    set_logger(name="", output_dir=args.output_path, rank=0, log_level=args.log_level)

    if args.task == "motion_transfer":
        inputs_config_path = "config/motion_transfer.yaml"
    elif args.task == "motion_style_transfer":
        inputs_config_path = "config/motion_style_transfer.yaml"
    else:
        raise ValueError(f"{args.task} is invalid, should be in [text2img, img2img, inpaint]")
    inputs_config = OmegaConf.load(inputs_config_path)
    print("D--: inputs_config")

    key_settings_info = ["Key Settings:\n" + "=" * 50]
    key_settings_info += [
        f"VC export task: {args.task}",
        f"inputs config: {inputs_config_path}",
        f"Number of samples in each trial: {args.n_samples}",
        f"Sampler: {args.sampler}",
    ]
    for key in inputs_config.keys():
        key_settings_info.append(f"{key}: {inputs_config[key]}")

    logger.info("\n".join(key_settings_info))

    args.inputs_config = inputs_config
    main(args)
