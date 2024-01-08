"""
AnimateDiff training pipeline
"""
import argparse
import ast
import datetime
import logging
import math
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

import mindspore as ms
from mindspore import context
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore import Model
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import LossMonitor, TimeMonitor

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed
from mindone.utils.params import count_params
from mindone.utils.version_control import is_910b
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.ema import EMA
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallback

from ad.utils.load_models import build_model_from_config, load_motion_modules
from ad.data.dataset import create_dataloader, check_sanity

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/training/mmv2_train.yaml",
        help="config path a yaml file defining model arch and training hyper-params",
    )
    # env args
    parser.add_argument("--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument("--is_parallel", type=ast.literal_eval, default=False, help="Set True for parallel training")
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument("--max_device_memory", type=str, default=None)
    parser.add_argument("--ms_enable_graph_kernel", type=ast.literal_eval, default=False, help="Enable graph op fusion for acceleration (on GPU)")
    # paths
    parser.add_argument("--output_dir", type=str, default="outputs/train", help="folder for saving checkpoints and logs")

    args = parser.parse_args()

    return args


# TODO: extract as mindone common api
def init_env(args):
    set_random_seed(args.seed)

    # Set Mindspore Context
    context.set_context(
            mode=args.ms_mode,
            device_target=args.device_target,
            pynative_synchronize=False,  # True for debug
            )
    device_id = 0  # TODO:
    if args.device_target == "Ascend":
        device_id = int(os.getenv("DEVICE_ID", 0))
        context.set_context(device_id=device_id)
    elif args.device_target == "GPU" and args.ms_enable_graph_kernel:
        context.set_context(enable_graph_kernel=True)

    if args.max_device_memory is not None:
        context.set_context(max_device_memory=args.max_device_memory)
        context.set_context(memory_optimize_level="O1", ascend_config={"atomic_clean_policy": 1})
    # context.set_context(precision_mode="allow_fp32_to_fp16")

    # Set Parallel
    if args.is_parallel:
        init()
        rank_id, rank_size, parallel_mode = get_rank(), get_group_size(), context.ParallelMode.DATA_PARALLEL
        ms.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=rank_size, parallel_mode=parallel_mode, gradients_mean=True)
    else:
        rank_id, rank_size = 0, 1
    
    # Set logger
    # time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    # save_dir = f"outputs/{Path(args.config).stem}-{time_str}"
    set_logger(name="", output_dir=args.output_dir)

    logger.info(f"Device_id: {device_id}, rank_id: {rank_id}, device_num: {rank_size}")

    return device_id, rank_id, rank_size


def train(args):
    # 1. init 
    device_id, rank_id, rank_size = init_env(args)
    cfg = OmegaConf.load(args.config)
    use_motion_module = not cfg.image_finetune
    logger.info("Task: {}".format("image finetuning" if cfg.image_finetune else "temperal module training"))

    # 2. build model and load pretrained weights
    # vae, text_encoder, unet3d, and the latent diffusion model with loss
    ldm = build_model_from_config(
        cfg,
        ckpt=cfg.pretrained_model_path,
        use_motion_module=use_motion_module,
        is_training=True,
    )
    unet = ldm.model
    # text_encoder = ldm.cond_stage_model
    # vae = ldm.first_stage_model

    # load motion module weights if provided
    if use_motion_module:
        if cfg.get("pretrained_motion_module_path", "") != "":
            load_motion_modules(ldm.model, cfg.pretrained_motion_module_path)
        # set motion module precision. TODO: will it lead to param prefix change?
        unet.diffusion_model = unet.diffusion_model.set_mm_amp_level(cfg.mm_amp_level)

    # 3. optional: inject motion lora for lora finetune
    
    # set trainable params and freeze the others
    # vae and clip are frozen by default
    if not cfg.image_finetune:
        # only motion moduel trainable
        num_mm_params = 0
        for param in unet.get_parameters():
            if '.temporal_transformer.' in param.name:
                param.requires_grad = True
                num_mm_params += 1
            else:
                param.requires_grad = False
        logger.info("Num MM params {}".format(num_mm_params))
    
    # count total params and trainable params
    tot_params, trainable_params = count_params(ldm.model)
    logger.info("UNet3D: total param size {:,}, trainable {:,}".format(tot_params, trainable_params)) 
    assert trainable_params > 0, "No trainable parameters. Please check model config."

    # 4. build dataset 
    tokenizer = ldm.cond_stage_model.tokenize
    dataloader = create_dataloader(
            cfg.train_data, 
            tokenizer=tokenizer, 
            is_image=cfg.image_finetune,
            device_num=rank_size,
            rank_id=rank_id)
    num_dataset_batches = dataloader.get_dataset_size()

    check_data = False # debug
    if check_data:
        iterator = dataloader.create_dict_iterator()
        for i, batch in enumerate(iterator):
            for k in batch:
                print(k, batch[k].shape, batch[k].dtype)
            check_sanity(batch['video'][0].asnumpy(), f"sanity{i}.gif")
            logger.info("save dataloader output sanity in "+f"sanity{i}.gif")
            break

    # 5. build training utilites
    # lr scheduler
    lr = create_scheduler(num_dataset_batches, **cfg.lr_schedule)

    # optimizer
    optimizer = create_optimizer(
        params=ldm.trainable_params(),
        name=cfg.optim.name,
        lr=lr,
        betas=cfg.optim.betas,
        weight_decay=cfg.optim.weight_decay,
        )

    # loss scaler
    if cfg.loss_scaler.type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=cfg.loss_scaler.loss_scale, scale_factor=cfg.loss_scaler.scale_factor, scale_window=cfg.loss_scaler.scale_window
        )
    elif cfg.loss_scaler.type == "static":
        loss_scaler = ms.nn.FixedLossScaleUpdateCell(cfg.loss_scaler.loss_scale)
    else:
        raise ValueError
    # ema
    ema = (
        EMA(
            ldm,
            trainable_only=True,
            ema_decay=cfg.ema_decay,
        )
        if cfg.use_ema
        else None
    )
    # train one step function
    net_with_grads = TrainOneStepWrapper(
        ldm,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=cfg.drop_overflow_update,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        clip_grad=cfg.clip_grad,
        clip_norm=cfg.max_grad_norm,
        ema=ema,
    )
    model = Model(net_with_grads)

    # callbacks
    callbacks = [TimeMonitor(cfg.log_interval), LossMonitor(cfg.log_interval), OverflowMonitor()]
    if cfg.profile:
        callbacks.append(ProfilerCallback())
    # TODO: support resume training from previous step
    # TODO: for step mode + data sink, ckpt_save_interval needs conversion according to sink_size.
    start_epoch = 0
    if rank_id == 0:
        model_name = 'ad_v2' if not cfg.image_finetune else 'sd_v1.5'
        # TODO: support saving motion module only; 
        save_cb = EvalSaveCallback(
            network=ldm,
            save_trainable_only=not cfg.save_whole_model, # debug
            rank_id=rank_id,
            ckpt_save_dir=args.output_dir,
            ema=ema,
            ckpt_save_policy="latest_k",
            ckpt_max_keep=cfg.ckpt_max_keep,
            step_mode=True,
            ckpt_save_interval=cfg.ckpt_save_interval,
            log_interval=cfg.log_interval,
            start_epoch=start_epoch,
            record_lr=not is_910b(),  # TODO: try record lr on 910b for newer MS version
            model_name=model_name,
        )
        callbacks.append(save_cb)
        
        # create folders and save config
        # TODO: save checkpoints to {output_dirs}/checkpoints/
        OmegaConf.save(cfg, os.path.join(args.output_dir, 'config.yaml'))
        

    # 6. launch training
    # compute training epochs
    assert not cfg.train_epochs == cfg.train_steps == -1 
    if cfg.train_steps == -1:
        epochs = cfg.train_epochs
    else:
        if cfg.sink_size == -1:
            epochs = math.ceil(cfg.train_steps / num_dataset_batches)
        else:
            epochs = math.ceil(cfg.train_steps / cfg.sink_size)

    logger.info(f"Training steps {cfg.train_steps} => epochs {epochs}") 
    logger.info("Start training. Please wait for graph compilation... (~20 min)")
    model.train(
        epochs, dataloader, callbacks=callbacks, dataset_sink_mode=cfg.dataset_sink_mode, sink_size=cfg.sink_size, initial_epoch=start_epoch,
    )


if __name__ == "__main__":
    args = parse_args()
    train(args)
