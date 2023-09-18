import argparse
import logging
import os
import sys
import time

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

import mindspore as ms

workspace = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(workspace))

sd_abs_path = os.path.dirname(os.path.dirname(workspace)) + "/stable_diffusion_v2"
sys.path.append(sd_abs_path)

from vc.data.dataset_lite_infer import load_data

from ldm.modules.logger import set_logger
from ldm.util import instantiate_from_config
from libs.helper import VaeImageProcessor
from libs.infer_engine.sd_lite_models import VCLiteMotionStyleTransfer 
from tools._common.clip import CLIPTokenizer

logger = logging.getLogger("Video Composer Lite Deploy")


def get_mindir_path(model_save_path, name):
    mindir_path = os.path.join(model_save_path, f"{name}_lite.mindir")
    if not os.path.exists(mindir_path):
        mindir_path = os.path.join(model_save_path, f"{name}_lite_graph.mindir")
    return mindir_path


def main(args):
    # init env
    set_logger(name="", output_dir=args.output_path, rank=0, log_level=args.log_level)
    ms.set_context(device_target="CPU")
    args.sample_path = os.path.join(args.output_path, "samples", args.task)
    os.makedirs(args.sample_path, exist_ok=True)
    args.base_count = len(os.listdir(args.sample_path))
    
    # 0. Load args and model config
    from config.model_config import cfg  # base arch config

    bs = batch_size = args.n_samples
    model_save_path = os.path.join(args.output_path, args.task)
    logger.info(f"model_save_path: {model_save_path}")

    # create sampler. only used to get type and timesteps  
    sampler_config = OmegaConf.load(args.sampler)
    _scheduler = instantiate_from_config(sampler_config)
    timesteps = _scheduler.set_timesteps(args.sampling_steps)
    scheduler_type = sampler_config.type
  
    data_prepare = get_mindir_path(model_save_path, args.inputs.data_prepare_model)
    scheduler_preprocess = get_mindir_path(model_save_path, f"{args.inputs.scheduler_preprocess}-{scheduler_type}")
    predict_noise = get_mindir_path(model_save_path, args.inputs.predict_noise_model)
    noisy_sample = get_mindir_path(model_save_path, f"{args.inputs.noisy_sample_model}-{scheduler_type}")
    vae_decoder = get_mindir_path(model_save_path, args.inputs.vae_decoder_model)

    # 1. Prepare inputs for pipeline sub-graphs

    # DataPrepare: prompt_data, negative_prompt_data, noise, style_image, single_image, motion_vectors
    # NoisePredict: -latents, +ts, -text_emb, -style_emb, -single_image_tr, -motion_vectors_tr, +guidance_scale):
    # SchedulerPreprocess: #latents, t
    # NoisySample: -noise_pred, #ts, #latents, +num_inference_steps):

    img_processor = VaeImageProcessor()
    tokenizer = CLIPTokenizer("./model_weights/bpe_simple_vocab_16e6.txt.gz")
    
    # the data type must be the same as defined in export.py
    inputs = load_data(
            cfg, 
            tokenizer, 
            video_path=args.inputs.input_video, 
            prompt=args.inputs.prompt,
            neg_prompt=args.inputs.negative_prompt,
            single_image_path=args.inputs.single_image,
            style_image_path=args.inputs.style_image,
            use_fp16=True,
            )

    for k in inputs:
        print("D--: ", k, inputs[k].shape)

    inputs["prompt"] = args.inputs.prompt
    inputs["negative_prompt"] = args.inputs.negative_prompt
    inputs["timesteps"] = timesteps
    inputs["scale"] = np.array(args.scale, np.float16)
    
    # 2. Create models and pipeline 
    if args.task == "motion_style_transfer":
        vc_infer = VCLiteMotionStyleTransfer(
            data_prepare,
            scheduler_preprocess,
            predict_noise,
            noisy_sample,
            vae_decoder,
            device_target=args.device_target,
            device_id=int(os.getenv("DEVICE_ID", 0)),
            num_inference_steps=args.sampling_steps,
        )
    else:
        raise ValueError(f"Not support task: {args.task}")

    for n in range(args.n_iter):
        start_time = time.time()
        inputs["noise"] = np.random.standard_normal(
            size=(batch_size, 4, cfg.max_frames, args.inputs.H // 8, args.inputs.W // 8)
        ).astype(np.float16)

        x_samples = vc_infer(inputs) # (b f 3 H W)
        x_samples = x_samples[0] # TODO: fix for batch size > 1 
        x_samples = img_processor.postprocess(x_samples) # -> PIL image list
        
        vid_save_dir = os.path.join(args.sample_path,  f"vid{args.base_count:05}")
        os.makedirs(vid_save_dir, exist_ok=True)
        for fidx, frame in enumerate(x_samples):
            frame.save(os.path.join(vid_save_dir, f"{fidx:03}.png"))
        args.base_count += 1
        # TODO: save as gif

        end_time = time.time()
        logger.info(
            "{}/{} images generated, time cost for current trial: {:.3f}s".format(
                batch_size * (n + 1), batch_size * args.n_iter, end_time - start_time
            )
        )

    logger.info(f"Done! All generated images are saved in: {args.sample_path}" f"\nEnjoy.")


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
    parser.add_argument("--model", type=str, default=None, help="path to config which constructs model.")
    parser.add_argument("--output_path", type=str, default="output", help="dir to write results to")
    parser.add_argument("--sampler", type=str, default="config/schedule/ddim.yaml", help="infer sampler yaml path")
    parser.add_argument("--sampling_steps", type=int, default=50, help="number of ddim sampling steps")
    parser.add_argument("--n_iter", type=int, default=1, help="number of iterations or trials. sample this often, ")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt in an iteration. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: "
        "eps = eps(x, uncond) + scale * (eps(x, cond) - eps(x, uncond)). "
        "Simplified: `uc + scale * (uc - prompt)`",
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
    inputs = OmegaConf.load(inputs_config_path)

    key_settings_info = ["Key Settings:\n" + "=" * 50]
    key_settings_info += [
        f"SD Lite infer task: {args.task}",
        f"inputs config: {inputs_config_path}",
        f"Number of trials for each prompt: {args.n_iter}",
        f"Number of samples in each trial: {args.n_samples}",
        f"Sampler: {args.sampler}",
        f"Sampling steps: {args.sampling_steps}",
        f"Uncondition guidance scale: {args.scale}",
    ]
    for key in inputs.keys():
        key_settings_info.append(f"{key}: {inputs[key]}")

    logger.info("\n".join(key_settings_info))

    args.inputs = inputs
    main(args)
