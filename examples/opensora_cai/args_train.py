import argparse
import logging
import os

import yaml
from opensora.utils.model_utils import _check_cfgs_in_parser, str2bool

logger = logging.getLogger()


def parse_train_args(parser):
    parser.add_argument(
        "--config",
        "-c",
        default="",
        type=str,
        help="path to load a config yaml file that describes the training recipes which will override the default arguments",
    )
    # the following args's defualt value will be overrided if specified in config yaml

    # data
    parser.add_argument("--dataset_name", default="", type=str, help="dataset name")
    parser.add_argument(
        "--csv_path",
        default="",
        type=str,
        help="path to csv annotation file. columns: video, caption. video indicates the relative path of video file in video_folder. caption - the text caption for video",
    )
    parser.add_argument("--video_column", default="video", type=str, help="name of column for videos saved in csv file")
    parser.add_argument(
        "--caption_column", default="caption", type=str, help="name of column for captions saved in csv file"
    )
    parser.add_argument("--video_folder", default="", type=str, help="root dir for the video data")
    parser.add_argument("--embed_folder", default="", type=str, help="root dir for the text embeding data")
    parser.add_argument("--output_path", default="output/", type=str, help="output directory to save training results")
    parser.add_argument(
        "--pretrained_model_path",
        default="",
        type=str,
        help="Specify the pretrained model path, either a pretrained " "DiT model or a pretrained Latte model.",
    )
    # model
    parser.add_argument("--space_scale", default=0.5, type=float, help="stdit model space scalec")
    parser.add_argument("--time_scale", default=1.0, type=float, help="stdit model time scalec")
    # ms
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--max_device_memory", type=str, default=None, help="e.g. `30GB` for 910a, `59GB` for 910b")
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--parallel_mode", default="data", type=str, choices=["data", "optim"], help="parallel mode: data, optim"
    )

    # modelarts
    parser.add_argument("--enable_modelarts", default=False, type=str2bool, help="run codes in ModelArts platform")
    parser.add_argument("--num_workers", default=1, type=int, help="the number of modelarts workers")
    parser.add_argument(
        "--json_data_path",
        default="mindone/examples/stable_diffusion_v2/ldm/data/num_samples_64_part.json",
        type=str,
        help="the path of num_samples.json containing a dictionary with 64 parts. "
        "Each part is a large dictionary containing counts of samples of 533 tar packages.",
    )
    parser.add_argument(
        "--resume",
        default=False,
        type=str,
        help="It can be a string for path to resume checkpoint, or a bool False for not resuming.(default=False)",
    )
    # training hyper-params
    parser.add_argument("--optim", default="adamw", type=str, help="optimizer")
    parser.add_argument(
        "--betas",
        type=float,
        default=[0.9, 0.999],
        help="Specify the [beta1, beta2] parameter for the AdamW optimizer.",
    )
    parser.add_argument(
        "--optim_eps", type=float, default=1e-6, help="Specify the eps parameter for the AdamW optimizer."
    )
    parser.add_argument(
        "--group_strategy",
        type=str,
        default="norm_and_bias",
        help="Grouping strategy for weight decay. If `norm_and_bias`, weight decay filter list is [beta, gamma, bias]. \
                If None, filter list is [layernorm, bias]. Default: norm_and_bias",
    )

    parser.add_argument("--weight_decay", default=1e-6, type=float, help="Weight decay.")
    parser.add_argument("--seed", default=3407, type=int, help="data path")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="warmup steps")
    parser.add_argument("--batch_size", default=10, type=int, help="batch size")
    parser.add_argument("--start_learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float, help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=0, type=int, help="lr decay steps.")
    parser.add_argument("--scheduler", default="cosine_decay", type=str, help="scheduler.")

    # dataloader params
    parser.add_argument("--dataset_sink_mode", default=False, type=str2bool, help="sink mode")
    parser.add_argument("--sink_size", default=-1, type=int, help="dataset sink size. If -1, sink size = dataset size.")
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        help="epochs. If dataset_sink_mode is on, epochs is with respect to dataset sink size. Otherwise, it's w.r.t the dataset size.",
    )
    parser.add_argument("--init_loss_scale", default=65536, type=float, help="loss scale")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=1000, type=float, help="scale window")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    # parser.add_argument("--cond_stage_trainable", default=False, type=str2bool, help="whether text encoder is trainable")
    parser.add_argument("--use_ema", default=False, type=str2bool, help="whether use EMA")
    parser.add_argument("--clip_grad", default=False, type=str2bool, help="whether apply gradient clipping")
    parser.add_argument(
        "--use_recompute",
        default=False,
        type=str2bool,
        help="whether use recompute.",
    )
    parser.add_argument(
        "--patch_embedder",
        type=str,
        default="conv",
        choices=["conv", "linear"],
        help="Whether to use conv2d layer or dense (linear layer) as Patch Embedder.",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--precision_mode",
        default=None,
        type=str,
        help="If specified, set the precision mode for Ascend configurations.",
    )
    parser.add_argument(
        "--model_name",
        "-m",
        type=str,
        default="Latte-XL/2",
        help="Model name , such as Latte-XL/2, Latte-L/2",
    )
    parser.add_argument("--t5_model_dir", default=None, type=str, help="the T5 cache folder path")
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="models/sd-vae-ft-mse.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument(
        "--sd_scale_factor", type=float, default=0.18215, help="VAE scale factor of Stable Diffusion model."
    )
    parser.add_argument("--image_size", default=256, type=int, help="the image size used to initiate model")
    parser.add_argument("--num_frames", default=16, type=int, help="the num of frames used to initiate model")
    parser.add_argument("--frame_stride", default=3, type=int, help="frame sampling stride")

    parser.add_argument(
        "--random_drop_text", default=False, type=str2bool, help="set caption to empty string randomly if enabled"
    )
    parser.add_argument(
        "--disable_flip",
        default=True,
        type=str2bool,
        help="disable random flip video (to avoid motion direction and text mismatch)",
    )
    parser.add_argument("--random_drop_text_ratio", default=0.1, type=float, help="drop ratio")
    parser.add_argument(
        "--enable_flash_attention",
        default=None,
        type=str2bool,
        help="whether to enable flash attention.",
    )
    parser.add_argument("--drop_overflow_update", default=True, type=str2bool, help="drop overflow update")
    parser.add_argument("--loss_scaler_type", default="dynamic", type=str, help="dynamic or static")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )

    parser.add_argument("--ckpt_save_interval", default=1, type=int, help="save checkpoint every this epochs or steps")
    parser.add_argument("--ckpt_max_keep", default=10, type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument(
        "--step_mode",
        default=False,
        type=str2bool,
        help="whether save ckpt by steps. If False, save ckpt by epochs.",
    )

    parser.add_argument("--profile", default=False, type=str2bool, help="Profile or not")
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    parser.add_argument("--log_interval", type=int, default=1, help="log interval")
    return parser


def parse_embedding_cache_args(parser):
    parser.add_argument(
        "--cache_file_type",
        default="mindrecord",
        type=str,
        choices=["numpy", "mindrecord"],
        help="type of cached dataset file",
    )
    parser.add_argument(
        "--save_data_type",
        default="float32",
        type=str,
        choices=["float16", "float32"],
        help="data type when saving embedding cache",
    )
    parser.add_argument("--cache_folder", default="", type=str, help="directory to save embedding cache")
    parser.add_argument(
        "--max_page_size",
        default=256,
        type=int,
        choices=[64, 128, 256],
        help="The maximum page size for the MindRecord File Writer. Should be one of [64, 128, 256]",
    )
    parser.add_argument(
        "--resume_cache_index", default=None, type=int, help="If provided, will resume cache from this video index."
    )
    parser.add_argument(
        "--dump_every_n_lines",
        type=int,
        default=1,
        help="The number of data items (videos) saved every time calling mindrecord writer.",
    )
    return parser


def parse_args():
    parser = argparse.ArgumentParser()
    parser = parse_train_args(parser)
    parser = parse_embedding_cache_args(parser)
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    default_args = parser.parse_args()
    if default_args.config:
        default_args.config = os.path.join(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()

    print(args)

    return args
