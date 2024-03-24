"""
Train AutoEncoder KL-reg with GAN loss
"""
import os
import sys
import argparse
import logging
import yaml
import time
import shutil
from omegaconf import OmegaConf
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore import Model, load_checkpoint, load_param_into_net, nn
from mindspore.train.callback import TimeMonitor

from ldm.util import str2bool
from ldm.models.autoencoder import GeneratorWithLoss
from ldm.data.dataset_vae import create_dataloader 
from ldm.util import instantiate_from_config

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
from mindone.env import init_train_env
from mindone.utils.logger import set_logger
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallback
from mindone.trainers.checkpoint import resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.params import count_params, load_param_into_net_with_filter
from mindone.trainers.checkpoint import CheckpointManager


os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"
os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "INFNAN_MODE"

logger = logging.getLogger(__name__)

def create_loss_scaler(loss_scaler_type, init_loss_scale, loss_scale_factor=2, scale_window=1000):
    if args.loss_scaler_type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=args.init_loss_scale, scale_factor=args.loss_scale_factor, scale_window=args.scale_window
        )
    elif args.loss_scaler_type == "static":
        loss_scaler = nn.FixedLossScaleUpdateCell(args.init_loss_scale)
    else:
        raise ValueError

    return loss_scaler 

def main(args):
    # 1. init
    # ascend_config={"precision_mode": "allow_fp32_to_fp16"}
    device_id, rank_id, device_num = init_train_env(
        args.mode,
        device_target=args.device_target,
        seed=args.seed,
        distributed=args.use_parallel,
    )
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # 2. build models
    ##  autoencoder (G)
    config = OmegaConf.load(args.base_config)
    # TODO: allow set bf16
    if args.dtype == 'fp32': 
        config.generator.params.use_fp16=False
    else:
        config.generator.params.use_fp16=True
    ae = instantiate_from_config(config.generator) 
    # TODO: allow loading pretrained weights

    ## discriminator (D)
    if args.use_discriminator:
        disc = instantiate_from_config(config.discriminator) 
    else:
        disc = None
    # TODO: allow loading pretrained weights for D
    
    # 3. build net with loss (core)
    ## G with loss
    ae_with_loss = GeneratorWithLoss(ae, discriminator=disc, **config.lossconfig)
    
    ## D with loss
    if args.use_discriminator:
        disc_with_loss = None
        raise NotImplementedError

    tot_params, trainable_params = count_params(ae_with_loss)
    logger.info("ae with loss: total param size {:,}, trainable {:,}".format(tot_params, trainable_params))
    trainable_ae = count_params(ae)[1]
    # TODO: check logvar trainability
    assert trainable_params <= trainable_ae+1, f"ae trainable: {trainable_ae}"

    # 4. build dataset
    ds_config = dict(
        csv_path=args.csv_path,
        image_folder=args.data_path,
        size=args.size,
        crop_size=args.crop_size,
        random_crop=args.random_crop,
        flip=args.flip,
        )
    dataset = create_dataloader(
        ds_config=ds_config, 
        batch_size=args.batch_size, 
        num_parallel_workers=args.num_parallel_workers,
        shuffle=args.shuffle, 
        device_num=device_num, 
        rank_id=rank_id)
    dataset_size = dataset.get_dataset_size()

    # 5. build training utils
    # torch scale lr by: model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr 
    if args.scale_lr:
        learning_rate = args.base_learning_rate * args.batch_size * args.gradient_accumulation_steps * device_num
    else:
        learning_rate = args.base_learning_rate

    # build optimizer
    # TODO: Adam optimizer in MS is less reliable?
    # TODO: optim hyper-params like beta copied from torch may not be suitable for MS
    update_logvar = False # in torch, ae_with_loss.logvar  is not updated.
    if update_logvar:
        params_to_update =  ae_with_loss.trainable_params()
    else:
        params_to_update =  ae_with_loss.autoencoder.trainable_params()
    optim_ae = create_optimizer(
        params_to_update,
        name=args.optim,
        betas=(0.5, 0.9),
        group_strategy=args.group_strategy,
        weight_decay=args.weight_decay,
        lr=learning_rate,
    )
    loss_scaler_ae = create_loss_scaler(args.loss_scaler_type, args.init_loss_scale, args.loss_scale_factor, args.scale_window)

    if args.use_discriminator:
        # TODO: different LR for disc?
        optim_disc = create_optimizer(
                disc.trainable_paramrs(), 
                betas=(0.5, 0.9),
                name=args.optim,
                lr=learning_rate, 
                group_strategy=args.group_strategy,
                weight_decay=args.weight_decay,
                )
        # TODO; can we use two loss scalers?
        loss_scaler_disc = create_loss_scaler(args.loss_scaler_type, args.init_loss_scale, args.loss_scale_factor, args.scale_window)
    
    # build training step
    training_step_ae = TrainOneStepWrapper(
        ae_with_loss,
        optimizer=optim_ae,
        scale_sense=loss_scaler_ae,
        drop_overflow_update=args.drop_overflow_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,  # TODO: check
        clip_norm=args.max_grad_norm,
        ema=None,
    )

    if args.use_discriminator:
        training_step_ae = TrainOneStepWrapper(
            disc_with_loss,
            optimizer=optim_disc,
            scale_sense=loss_scaler_disc,
            drop_overflow_update=args.drop_overflow_update,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            clip_grad=args.clip_grad,  # TODO: check
            clip_norm=args.max_grad_norm,
            ema=None,
        )

    if rank_id == 0:
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"Distributed mode: {args.use_parallel}",
                f"Data path: {args.data_path}",
                f"Learning rate: {learning_rate}",
                f"Batch size: {args.batch_size}",
                f"Rescale size: {args.size}",
                f"Crop size: {args.crop_size}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num epochs: {args.epochs}",
                f"Loss scaler: {args.loss_scaler_type}",
                f"Init loss scale: {args.init_loss_scale}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

    # 6. training pipeline 
    start_epoch = 0
    if rank_id == 0:
        ckpt_dir = os.path.join(args.output_path, "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)

    ## Phase 1: only train ae. build training pipeline with model.train
    if not args.use_discriminator:
        use_custom_train = True
        if use_custom_train:
            # self-define train pipeline
            # TODO: support data sink, refer to mindspore/train/dataset_helper.py and mindspore.train.model
            # TODO: support step mode, resume train in step unit, dat
            ckpt_manager = CheckpointManager(ckpt_dir, "latest_k", k=args.ckpt_max_keep)
            ds_iter = dataset.create_dict_iterator(args.epochs - start_epoch)

            for epoch in range(start_epoch, args.epochs):
                start_time_e = time.time()
                # output_numpy=True ?
                for step, data in enumerate(ds_iter):
                    start_time_s = time.time()
                    x = data['image']

                    loss_ae_t, overflow, scaling_sens = training_step_ae(x)

                    cur_global_step = epoch * dataset_size + step + 1
                    if overflow:
                        logger.warning(f"Overflow occurs in step {cur_global_step}")

                    loss_ae = float(loss_ae_t.asnumpy())

                    step_time = time.time() - start_time_s
                    if step % args.log_interval == 0:
                        logger.info(f"Loss ae: {loss_ae:.4f}, Step time: {step_time*1000:.2f}ms")
                epoch_cost = time.time() - start_time_e
                per_step_time = epoch_cost / dataset_size 
                cur_epoch = epoch + 1
                logger.info(f"Epoch:[{int(cur_epoch):>3d}/{int(args.epochs):>3d}], "
                    f"epoch time:{epoch_cost:.2f}s, per step time:{per_step_time*1000:.2f}ms, ")
                if (cur_epoch % args.ckpt_save_interval == 0) or (cur_epoch == args.epochs):
                    ckpt_name = f"vae_kl_f8-e{cur_epoch}.ckpt"
                    # TODO: set append_dict to save loss scale and data iteratered.
                    # TODO: save logvar if udpate it
                    ckpt_manager.save(ae, None, ckpt_name=ckpt_name, append_dict=None)

        else:
            model = Model(training_step_ae)

            # callbacks
            callback = [TimeMonitor(args.log_interval)]
            ofm_cb = OverflowMonitor()
            callback.append(ofm_cb)

            if rank_id == 0:
                save_cb = EvalSaveCallback(
                    network=ae_with_loss.autoencoder,
                    rank_id=rank_id,
                    ckpt_save_dir=ckpt_dir,
                    ema=None,
                    ckpt_save_policy="latest_k",
                    ckpt_max_keep=args.ckpt_max_keep,
                    ckpt_save_interval=args.ckpt_save_interval,
                    log_interval=args.log_interval,
                    start_epoch=start_epoch,
                    model_name="vae_kl_f8",
                    record_lr=False,
                )
                callback.append(save_cb)
                if args.profile:
                    callback.append(ProfilerCallback())

                logger.info("Start training...")
                # backup config files
                shutil.copyfile(args.base_config, os.path.join(args.output_path, os.path.basename(args.base_config)))

                with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
                    yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

            model.train(
                args.epochs,
                dataset,
                callbacks=callback,
                dataset_sink_mode=args.dataset_sink_mode,
                # sink_size=args.sink_size,
                initial_epoch=start_epoch,
            )
    else:
        raise NotImplementedError


def _check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser):
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    cfg_keys = list(cfgs.keys())
    for k in cfg_keys:
        if k not in actions_dest and k not in defaults_key:
            # raise KeyError(f"{k} does not exist in ArgumentParser!")
            cfgs.pop(k)
    return cfgs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_config",
        default="configs/train/autoencoder_kl_f8.yaml",
        type=str,
        help="base train config path to load a yaml file that override the default arguments",
    )
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument("--output_path", default="output/", type=str, help="output directory to save training results")
    parser.add_argument(
        "--resume",
        default=False,
        type=str,
        help="resume training, can set True or path to resume checkpoint.(default=False)",
    )
    # ms
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    
    parser.add_argument("--profile", default=False, type=str2bool, help="Profile or not")
    # data
    parser.add_argument("--data_path", default="dataset", type=str, help="data path")
    parser.add_argument("--csv_path", default=None, type=str, help="path to csv annotation file")
    parser.add_argument("--dataset_sink_mode", default=False, type=str2bool, help="sink mode")
    parser.add_argument("--shuffle", default=True, type=str2bool, help="data shuffle")
    parser.add_argument("--num_parallel_workers", default=12, type=int, help="num workers for data loading")
    parser.add_argument("--size", default=384, type=int, help="image rescale size")
    parser.add_argument("--crop_size", default=256, type=int, help="image crop size")
    parser.add_argument("--random_crop", default=False, type=str2bool, help="random crop for data augmentation")
    parser.add_argument("--flip", default=False, type=str2bool, help="horizontal flip for data augmentation")

    # optim
    parser.add_argument("--use_discriminator", default=False, type=str2bool, help="Phase 1 training does not use discriminator, set False to reduce memory cost in graph mode.")
    parser.add_argument("--dtype", default="fp32", type=str, choices=['fp32', 'fp16', 'bf16'], help="data type for mixed precision")
    parser.add_argument("--optim", default="adam", type=str, help="optimizer")
    parser.add_argument("--weight_decay", default=0., type=float, help="Weight decay.")
    parser.add_argument(
        "--group_strategy",
        type=str,
        default="norm_and_bias",
        help="Grouping strategy for weight decay. If `norm_and_bias`, weight decay filter list is [beta, gamma, bias]. \
                If None, filter list is [layernorm, bias]. Default: norm_and_bias",
    )
    parser.add_argument("--seed", default=3407, type=int, help="data path")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="warmup steps")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--log_interval", default=1, type=int, help="log interval")
    parser.add_argument("--base_learning_rate", default=4.5e-06, type=float, help="base learning rate, can be scaled by global batch size")
    parser.add_argument("--scale_lr", default=True, type=str2bool, help="scale base-lr by ngpu * batch_size * n_accumulate")
    parser.add_argument("--decay_steps", default=0, type=int, help="lr decay steps.")
    parser.add_argument("--scheduler", default="cosine_decay", type=str, help="scheduler. option: constant, cosine_decay, ")
    parser.add_argument("--epochs", default=10, type=int, help="epochs")
    parser.add_argument("--loss_scaler_type", default="static", type=str, help="dynamic or static")
    parser.add_argument("--init_loss_scale", default=1024, type=float, help="loss scale")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=1000, type=float, help="scale window")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--use_ema", default=False, type=str2bool, help="whether use EMA")
    parser.add_argument("--clip_grad", default=False, type=str2bool, help="whether apply gradient clipping")
    parser.add_argument("--drop_overflow_update", default=True, type=str2bool, help="drop overflow update")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )
    parser.add_argument("--ckpt_max_keep", default=10, type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument("--ckpt_save_interval", default=1, type=int, help="save checkpoint every this epochs or steps")
    parser.add_argument(
        "--step_mode",
        default=False,
        type=str2bool,
        help="whether save ckpt by steps. If False, save ckpt by epochs.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )

    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    default_args = parser.parse_args()
    if default_args.base_config:
        default_args.base_config = os.path.join(abs_path, default_args.base_config)
        with open(default_args.base_config, "r") as f:
            cfg = yaml.safe_load(f)
            cfg = _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    # args.model_config = os.path.join(abs_path, args.model_config)

    logger.info(args)
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
