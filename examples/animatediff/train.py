"""
AnimateDiff training pipeline 
- Image finetuning
- Motion module training 
"""
import sys
import importlib
import logging
import os
import shutil
import datetime
import yaml

from args_train import parse_args

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../stable_diffusion_v2/")))

# TODO: use API in mindone
from common import init_env
from ldm.modules.logger import set_logger
from ldm.modules.train.callback import EvalSaveCallback, OverflowMonitor
from ldm.modules.train.checkpoint import resume_train_network
from ldm.modules.train.ema import EMA
from ldm.modules.train.lr_schedule import create_scheduler
from ldm.modules.train.optim import build_optimizer
from ldm.modules.train.trainer import TrainOneStepWrapper
from ldm.util import count_params, is_old_ms_version, str2bool
from omegaconf import OmegaConf

# from mindone.trainers.optim import create_optimizer
# from mindone.trainers.train_step import TrainOneStepWrapper

from mindspore import Model, Profiler, load_checkpoint, load_param_into_net, nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

from ad.data.dataset import create_dataloader, check_sanity
from ad.utils.load_models import update_unet2d_params_for_unet3d

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"

logger = logging.getLogger(__name__)


def build_model_from_config(config, enable_flash_attention=None):
    config = OmegaConf.load(config).model
    if args is not None:
        if enable_flash_attention is not None:
            config["params"]["unet_config"]["params"]["enable_flash_attention"] = enable_flash_attention
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    config_params = config.get("params", dict())
    # config_params['cond_stage_trainable'] = cond_stage_trainable # TODO: easy config
    return get_obj_from_str(config["target"])(**config_params)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_pretrained_model(pretrained_ckpt, net, unet_initialize_random=False, load_unet3d_from_2d=False):
    logger.info(f"Loading pretrained model from {pretrained_ckpt}")
    if os.path.exists(pretrained_ckpt):
        param_dict = load_checkpoint(pretrained_ckpt)

        if load_unet3d_from_2d:
            param_dict = update_unet2d_params_for_unet3d(param_dict)

        if unet_initialize_random:
            pnames = list(param_dict.keys())
            # pop unet params from pretrained weight
            for pname in pnames:
                if pname.startswith("model.diffusion_model"):
                    param_dict.pop(pname)
            logger.warning("UNet will be initialized randomly")

        if is_old_ms_version():
            param_not_load = load_param_into_net(net, param_dict)
        else:
            param_not_load, ckpt_not_load = load_param_into_net(net, param_dict)
        logger.info("Net params not load: {}, Total net params not loaded: {}".format(param_not_load, len(param_not_load)))
        logger.info("Ckpt params not load: {}, Total ckpt params not loaded: {}".format(ckpt_not_load, len(ckpt_not_load)))
    else:
        logger.warning(f"Checkpoint file {pretrained_ckpt} dose not exist!!!")


def load_pretrained_model_vae_unet_cnclip(pretrained_ckpt, cnclip_ckpt, net):
    new_param_dict = {}
    logger.info(f"Loading pretrained model from {pretrained_ckpt}, {cnclip_ckpt}")
    if os.path.exists(pretrained_ckpt) and os.path.exists(cnclip_ckpt):
        param_dict = load_checkpoint(pretrained_ckpt)
        cnclip_param_dict = load_checkpoint(pretrained_ckpt)
        for key in param_dict:
            if key.startswith("first") or key.startswith("model"):
                new_param_dict[key] = param_dict[key]
        for key in cnclip_param_dict:
            new_param_dict[key] = cnclip_param_dict[key]
        param_not_load = load_param_into_net(net, new_param_dict)
        logger.info("Params not load: {}".format(param_not_load))
    else:
        logger.warning(f"Checkpoint file {pretrained_ckpt}, {cnclip_ckpt} dose not exist!!!")


def main(args):
    if args.profile:
        # TODO: use profiler callback
        profiler = Profiler(output_path="./profiler_data")
        args.epochs = 3
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    args.output_path = os.path.join(args.output_path, time_str) 

    # 1. init
    device_id, rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        enable_modelarts=args.enable_modelarts,
        num_workers=args.num_workers,
        json_data_path=args.json_data_path,
    )
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # 2. build model
    latent_diffusion_with_loss = build_model_from_config(args.model_config, args.enable_flash_attention)
    # load sd pretrained weight
    load_pretrained_model(
        args.pretrained_model_path, latent_diffusion_with_loss, unet_initialize_random=args.unet_initialize_random, load_unet3d_from_2d=(not args.image_finetune),
    )

    # set motion module amp O2 if required for memory reduction
    if (not args.image_finetune) and (args.force_motion_module_amp_O2):
        logger.warning("Force to set motion module in amp level O2")
        latent_diffusion_with_loss.model.diffusion_model.set_mm_amp_level("O2")

    # set only motion module trainable if not image finetune
    if not args.image_finetune:
        # only motion moduel trainable
        num_mm_trainable = 0
        for param in latent_diffusion_with_loss.model.get_parameters():
            # exclude positional embedding params from training
            if ('.temporal_transformer.' in param.name) and ('.pe' not in param.name):
                param.requires_grad = True
                num_mm_trainable += 1
            else:
                param.requires_grad = False
        logger.info("Num MM trainable params {}".format(num_mm_trainable))
        # assert num_mm_trainable in [546, 520], "Expect 546 trainable params for MM-v2 or 520 for MM-v1."
    
    # count total params and trainable params
    tot_params, trainable_params = count_params(latent_diffusion_with_loss.model)
    logger.info("UNet3D: total param size {:,}, trainable {:,}".format(tot_params, trainable_params)) 
    assert trainable_params > 0, "No trainable parameters. Please check model config."

    # 3. build dataset
    if args.image_finetune:
        logger.info("Task is image finetune, num_frames and frame_stride is forced to 1")
        args.num_frames = 1
        args.frame_stride = 1
        data_config = dict(video_folder=args.data_path, csv_path=args.data_path+'/video_caption.csv', sample_size=args.image_size, sample_stride=args.frame_stride, sample_n_frames=args.num_frames, batch_size=args.train_batch_size, shuffle=True, num_parallel_workers=args.num_parallel_workers, max_rowsize=32)
    else:
        data_config = dict(video_folder=args.data_path, csv_path=args.data_path+'/video_caption.csv', sample_size=args.image_size, sample_stride=args.frame_stride, sample_n_frames=args.num_frames, batch_size=args.train_batch_size, shuffle=True, num_parallel_workers=args.num_parallel_workers, max_rowsize=64)

    tokenizer = latent_diffusion_with_loss.cond_stage_model.tokenize
    dataset = create_dataloader(
            data_config, 
            tokenizer=tokenizer, 
            is_image=args.image_finetune,
            device_num=device_num,
            rank_id=rank_id)
    dataset_size = dataset.get_dataset_size()


    # 4. build training utils: lr, optim, callbacks, trainer
    # build learning rate scheduler
    if not args.decay_steps:
        args.decay_steps = args.epochs * dataset_size - args.warmup_steps  # fix lr scheduling
        if args.decay_steps <= 0:
            logger.warning(
                f"decay_steps is {args.decay_steps}, please check epochs, dataset_size and warmup_steps. "
                f"Will force decay_steps to be set to 1."
            )
            args.decay_steps = 1

    lr = create_scheduler(
        steps_per_epoch=dataset_size,
        scheduler=args.scheduler,
        lr=args.start_learning_rate,
        min_lr=args.end_learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        num_epochs=args.epochs,
    )

    # build optimizer
    optimizer = build_optimizer(
        model=latent_diffusion_with_loss,
        name=args.optim,
        betas=args.betas,
        eps=args.optim_eps,
        group_strategy=args.group_strategy,
        weight_decay=args.weight_decay,
        lr=lr,    
    )
    '''
    optimizer = create_optimizer(
        latent_diffusion_with_loss.trainable_params(),
        name=args.optim,
        betas=args.betas,
        eps=args.optim_eps,
        group_strategy=args.group_strategy,
        weight_decay=args.weight_decay,
        lr=lr,
    )
    '''

    if args.loss_scaler_type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=args.init_loss_scale, scale_factor=args.loss_scale_factor, scale_window=args.scale_window
        )
    elif args.loss_scaler_type == "static":
        loss_scaler = nn.FixedLossScaleUpdateCell(args.init_loss_scale)
    else:
        raise ValueError

    # resume ckpt
    ckpt_dir = os.path.join(args.output_path, "ckpt")
    start_epoch = 0
    if args.resume:
        resume_ckpt = os.path.join(ckpt_dir, "train_resume.ckpt") if isinstance(args.resume, bool) else args.resume

        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(
            latent_diffusion_with_loss, optimizer, resume_ckpt
        )
        loss_scaler.loss_scale_value = loss_scale
        loss_scaler.cur_iter = cur_iter
        loss_scaler.last_overflow_iter = last_overflow_iter

    # trainer (standalone and distributed)
    ema = (
        EMA(
            latent_diffusion_with_loss,
            ema_decay=0.9999,
        )
        if args.use_ema
        else None
    )

    net_with_grads = TrainOneStepWrapper(
        latent_diffusion_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=args.drop_overflow_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )

    model = Model(net_with_grads)

    # callbacks
    callback = [TimeMonitor(args.callback_size)]
    ofm_cb = OverflowMonitor()
    callback.append(ofm_cb)

    if rank_id == 0:
        save_cb = EvalSaveCallback(
            network=latent_diffusion_with_loss,
            rank_id=rank_id,
            ckpt_save_dir=ckpt_dir,
            ema=ema,
            ckpt_save_policy="latest_k",
            ckpt_max_keep=10,
            step_mode=args.step_mode,
            ckpt_save_interval=args.ckpt_save_interval,
            log_interval=args.callback_size,
            start_epoch=start_epoch,
            model_name="sd" if args.image_finetune else "ad",
            param_save_filter=['.temporal_transformer.'] if args.save_mm_only else None,
            record_lr=False,  # TODO: check LR retrival for new MS on 910b 
        )
        callback.append(save_cb)
        # if args.profile:
        #     callbacks.append(ProfilerCallback())

    # 5. log and save config
    if rank_id == 0:
        num_params_unet, _ = count_params(latent_diffusion_with_loss.model.diffusion_model)
        num_params_text_encoder, _ = count_params(latent_diffusion_with_loss.cond_stage_model)
        num_params_vae, _ = count_params(latent_diffusion_with_loss.first_stage_model)
        num_params, num_trainable_params = count_params(latent_diffusion_with_loss)

        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"Distributed mode: {args.use_parallel}",
                f"Data path: {args.data_path}",
                f"Num params: {num_params:,} (unet: {num_params_unet:,}, text encoder: {num_params_text_encoder:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_trainable_params:,}",
                f"Precision: {latent_diffusion_with_loss.model.diffusion_model.dtype}",
                f"Learning rate: {args.start_learning_rate}",
                f"Batch size: {args.train_batch_size}",
                f"Image size: {args.image_size}",
                f"Frames: {args.num_frames}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num epochs: {args.epochs}",
                f"Loss scaler: {args.loss_scaler_type}",
                f"Init loss scale: {args.init_loss_scale}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
                f"Enable flash attention: {args.enable_flash_attention}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")
        # backup config files
        shutil.copyfile(args.model_config, os.path.join(args.output_path, os.path.basename(args.model_config)))

        with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

    # 6. train
    model.train(
        args.epochs, dataset, callbacks=callback, dataset_sink_mode=args.dataset_sink_mode, sink_size=args.sink_size, initial_epoch=start_epoch
    )

    if args.profile:
        profiler.analyse()


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args()
    main(args)
