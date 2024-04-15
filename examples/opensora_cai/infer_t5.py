import argparse
import datetime
import logging
import os
import sys
import time
import numpy as np

import yaml
from opensora.models.text_encoders import get_text_encoder_and_tokenizer
from opensora.pipelines import InferPipeline
from opensora.utils.model_utils import _check_cfgs_in_parser, count_params, remove_pname_prefix, str2bool

import mindspore as ms
from mindspore import Tensor, ops

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed
from mindone.visualize.videos import save_videos

logger = logging.getLogger(__name__)

skip_vae = True

def init_env(args):
    # no parallel mode currently
    ms.set_context(mode=args.mode)  # needed for MS2.0
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=args.mode,
        device_target=args.device_target,
        device_id=device_id,
    )
    if args.precision_mode is not None:
        ms.set_context(ascend_config={"precision_mode": args.precision_mode})
    return device_id


def main(args):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = f"samples/{time_str}"
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    init_env(args)
    set_random_seed(args.seed)

    # 2. model initiate and weight loading
    ckpt_path = args.t5_cache_folder
    text_encoder, tokenizer = get_text_encoder_and_tokenizer('t5', ckpt_path)
    text_encoder.set_train(False)
    for param in text_encoder.get_parameters():  # freeze latte_model
        param.requires_grad = False

    # init inputs
    logger.info(f"Sampling {len(args.captions)} caption")
    start_time = time.time()

    # infer
    text_tokens, mask = text_encoder.get_text_tokens_and_mask(args.captions, return_tensor=True)
    logger.info(f"Num tokens: {mask.asnumpy().sum(1)}") 

    # text_emb = ops.stop_gradient(text_encoder(text_tokens, mask))
    text_emb = text_encoder(text_tokens, mask)

    end_time = time.time()
    
    # save result
    np.savez(args.output_path, tokens=text_tokens.asnumpy(), mask=mask.asnumpy(), text_emb=text_emb.asnumpy())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="",
        type=str,
        help="path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument("--t5_cache_folder", default=None, type=str, help="the T5 cache folder path")
    # MS new args
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    parser.add_argument(
        "--enable_flash_attention",
        default=False,
        type=str2bool,
        help="whether to enable flash attention. Default is False",
    )
    parser.add_argument(
        "--dtype",
        default="fp32",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp32`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--precision_mode",
        default=None,
        type=str,
        help="If specified, set the precision mode for Ascend configurations.",
    )
    parser.add_argument(
        "--use_recompute",
        default=False,
        type=str2bool,
        help="whether use recompute.",
    )
    parser.add_argument(
        "--captions",
        type=str,
        nargs="+",
        help="A list of text captions to be generated with",
    )
    parser.add_argument("--output_path", type=str, default="outputs/t5_embed.npz", help="path to save t5 embedding")

    default_args = parser.parse_args()
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = os.path.join(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            # _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**dict(
                captions=cfg['captions'],
                t5_cache_folder=cfg['t5_cache_folder'],
                ))
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
    
