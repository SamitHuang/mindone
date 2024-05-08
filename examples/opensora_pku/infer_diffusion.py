import argparse
import datetime
import logging
import os
import sys
import time
from typing import Tuple

import numpy as np
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm
from utils.model_utils import _check_cfgs_in_parser

import mindspore as ms
from mindspore import nn
from mindspore.communication.management import get_group_size, get_rank, init

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

import pandas as pd
from opensora.data.text_dataset import create_dataloader
from opensora.models.diffusion.latte_t2v import Attention, LatteT2V, LayerNorm
from opensora.text_encoders.t5_embedder import T5Embedder

from examples.opensora_pku.opensora.pipelines import VideoGenPipeline
from mindone.diffusers.schedulers import DDIMScheduler, DDPMScheduler
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import instantiate_from_config, str2bool
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params
from mindone.utils.seed import set_random_seed
from mindone.visualize.videos import save_videos

logger = logging.getLogger(__name__)


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    distributed: bool = False,
    max_device_memory: str = None,
    device_target: str = "Ascend",
    parallel_mode: str = "data",
    enable_dvm: bool = False,
    precision_mode: str = None,
) -> Tuple[int, int, int]:
    """
    Initialize MindSpore environment.

    Args:
        mode: MindSpore execution mode. Default is 0 (ms.GRAPH_MODE).
        seed: The seed value for reproducibility. Default is 42.
        distributed: Whether to enable distributed training. Default is False.
    Returns:
        A tuple containing the device ID, rank ID and number of devices.
    """
    set_random_seed(seed)

    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)

    if distributed:
        ms.set_context(
            mode=mode,
            device_target=device_target,
        )
        if parallel_mode == "optim":
            print("use optim parallel")
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
                enable_parallel_optimizer=True,
            )
            init()
            device_num = get_group_size()
            rank_id = get_rank()
        else:
            init()
            device_num = get_group_size()
            rank_id = get_rank()
            logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
            ms.reset_auto_parallel_context()

            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
                device_num=device_num,
            )

        var_info = ["device_num", "rank_id", "device_num / 8", "rank_id / 8"]
        var_value = [device_num, rank_id, int(device_num / 8), int(rank_id / 8)]
        logger.info(dict(zip(var_info, var_value)))

    else:
        device_num = 1
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
        )

    if enable_dvm:
        print("enable dvm")
        ms.set_context(enable_graph_kernel=True)
    if precision_mode is not None and len(precision_mode) > 0:
        ms.set_context(ascend_config={"precision_mode": precision_mode})
    return rank_id, device_num


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="",
        type=str,
        help="path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument(
        "--vae_config",
        type=str,
        default="configs/ae/causal_vae_488.yaml",
        help="path to load a config yaml file that describes the VAE model",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="image size in [256, 512]",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1000,
        help="number of classes, applies only when condition is `class`",
    )

    parser.add_argument("--batch_size", default=1, type=int, help="batch size for dataloader")
    parser.add_argument(
        "--model_version",
        type=str,
        default="17x256x256",
        help="Model version in ['17x256x256', '65x256x256', '65x512x512'] ",
    )
    parser.add_argument(
        "--condition",
        default=None,
        type=str,
        help="the condition types: `None` means using no conditions; `text` means using text embedding as conditions;"
        " `class` means using class labels as conditions.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="latte checkpoint path. If specified, will load from it, otherwise, will use random initialization",
    )
    parser.add_argument(
        "--text_encoder",
        default=None,
        type=str,
        choices=["clip", "t5"],
        help="text encoder for extract text embeddings: clip text encoder or t5-v1_1-xxl.",
    )
    parser.add_argument("--t5_max_length", type=int, default=120, help="the max length for the tokens")
    parser.add_argument("--t5_cache_folder", default=None, type=str, help="the T5 cache folder path")
    parser.add_argument(
        "--clip_checkpoint",
        type=str,
        default=None,
        help="CLIP text encoder checkpoint (or sd checkpoint to only load the text encoder part.)",
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="models/ae/causal_vae_488.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument(
        "--sd_scale_factor", type=float, default=0.18215, help="VAE scale factor of Stable Diffusion model."
    )

    parser.add_argument("--sampling_steps", type=int, default=50, help="Diffusion Sampling Steps")
    parser.add_argument("--guidance_scale", type=float, default=8.5, help="the scale for classifier-free guidance")
    # MS new args
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--max_device_memory", type=str, default=None, help="e.g. `30GB` for 910a, `59GB` for 910b")
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--parallel_mode", default="data", type=str, choices=["data", "optim"], help="parallel mode: data, optim"
    )
    parser.add_argument("--enable_dvm", default=False, type=str2bool, help="enable dvm mode")

    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    parser.add_argument(
        "--enable_flash_attention",
        default=False,
        type=str2bool,
        help="whether to enable flash attention. Default is False",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--amp_level", type=str, default="O2", help="Set the amp level for the transformer model. Defaults to O2."
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
        "--patch_embedder",
        type=str,
        default="conv",
        choices=["conv", "linear"],
        help="Whether to use conv2d layer or dense (linear layer) as Patch Embedder.",
    )
    parser.add_argument(
        "--captions",
        type=str,
        nargs="+",
        help="A list of text captions to be generated with",
    )
    parser.add_argument(
        "--num_videos_per_prompt", type=int, default=1, help="the number of images to be generated for each prompt"
    )
    parser.add_argument("--ddim_sampling", type=str2bool, default=True, help="Whether to use DDIM for sampling")
    parser.add_argument(
        "--save_latents",
        action="store_true",
        help="Whether to save latents (before vae decoding) instead of video files.",
    )
    parser.add_argument(
        "--decode_latents",
        action="store_true",
        help="whether to load the existing latents saved in npy files and run vae decoding",
    )
    parser.add_argument(
        "--input_latents_dir",
        type=str,
        default="",
        help="the directory where the latents in npy files are saved in. Only works when decode_latents is True.",
    )
    default_args = parser.parse_args()
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = os.path.join(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = f"samples/{time_str}"
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    # 1. init env
    args = parse_args()
    # 1. init
    rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device_target,
        max_device_memory=args.max_device_memory,
        parallel_mode=args.parallel_mode,
        enable_dvm=args.enable_dvm,
        precision_mode=args.precision_mode,
    )
    if args.precision_mode == "allow_fp32_to_fp16":
        logger.warning(f"T5 model may produce wrong results under {args.precision_mode} precision mode!")
    # 2. model initiate and weight loading
    # 2.1 latte
    logger.info(f"Latte-{args.model_version} init")

    assert args.condition == "text", "LatteT2V only support text condition now!"
    assert args.text_encoder == "t5", "LatteT2V only support t5 text encoder now!"

    latte_model = LatteT2V.from_pretrained_2d(
        "models",
        subfolder=args.model_version,
        enable_flash_attention=args.enable_flash_attention,
        use_recompute=args.use_recompute,
    )
    # mixed precision
    if args.dtype == "fp32":
        model_dtype = ms.float32
    else:
        model_dtype = {"fp16": ms.float16, "bf16": ms.bfloat16}[args.dtype]
        latte_model = auto_mixed_precision(
            latte_model,
            amp_level=args.amp_level,
            dtype=model_dtype,
            custom_fp32_cells=[LayerNorm, Attention, nn.SiLU],
        )

    video_length, image_size = latte_model.config.video_length, args.image_size
    # latent_size = (image_size // ae_stride_config[args.ae][1], image_size // ae_stride_config[args.ae][2])
    latent_size = args.image_size // 8

    if len(args.checkpoint) > 0:
        logger.info(f"Loading ckpt {args.checkpoint} into LatteT2V")
        latte_model.load_from_checkpoint(args.checkpoint)
    else:
        logger.warning("Latte uses random initialization!")

    latte_model = latte_model.set_train(False)
    for param in latte_model.get_parameters():  # freeze latte_model
        param.requires_grad = False

    # 2.2 vae
    logger.info("vae init")
    config = OmegaConf.load(args.vae_config)
    vae = instantiate_from_config(config.generator)
    vae.init_from_ckpt(args.vae_checkpoint)
    vae.set_train(False)

    vae = auto_mixed_precision(vae, amp_level="O2", dtype=ms.float16)
    logger.info("Use amp level O2 for causal 3D VAE.")

    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False
    vae.latent_size = (latent_size, latent_size)

    # parse the caption input
    if not isinstance(args.captions, list):
        args.captions = [args.captions]
    # if input is a text file, where each line is a caption, load it into a list
    if len(args.captions) == 1 and args.captions[0].endswith("txt"):
        captions = open(args.captions[0], "r").readlines()
        args.captions = [i.strip() for i in captions]
    if len(args.captions) == 1 and args.captions[0].endswith("csv"):
        captions = pd.read_csv(args.captions[0])
        args.captions = [i.strip() for i in captions["cap"]]
    n = len(args.captions)
    assert n > 0, "No captions provided"
    logger.info(f"Number of prompts: {n}")
    logger.info(f"Number of generated samples for each prompt {args.num_videos_per_prompt}")

    # create dataloader for the captions
    csv_file = {"path": [], "cap": []}
    for i in range(n):
        for i_video in range(args.num_videos_per_prompt):
            ext = ".npy" if (args.save_latents or args.decode_latents) else ".gif"
            csv_file["path"].append(f"{i_video}-{args.captions[i].strip()[:100]}{ext}")
            csv_file["cap"].append(args.captions[i])
    temp_dataset_csv = os.path.join(save_dir, "dataset.csv")
    pd.DataFrame.from_dict(csv_file).to_csv(temp_dataset_csv, index=False, columns=csv_file.keys())

    ds_config = dict(
        data_file_path=temp_dataset_csv,
        tokenizer=None,  # tokenizer,
        video_column="path",
        caption_column="cap",
    )
    dataset = create_dataloader(
        ds_config,
        args.batch_size,
        ds_name="text",
        num_parallel_workers=12,
        max_rowsize=32,
        shuffle=False,  # be in order
        device_num=device_num,
        rank_id=rank_id,
        drop_remainder=False,
    )
    dataset_size = dataset.get_dataset_size()
    logger.info(f"Num batches: {dataset_size}")
    ds_iter = dataset.create_dict_iterator(1, output_numpy=True)

    if args.decode_latents:
        for step, data in tqdm(enumerate(ds_iter), total=dataset_size):
            file_paths = data["file_path"]
            loaded_latents = []
            for i_sample in range(args.batch_size):
                save_fp = os.path.join(args.input_latents_dir, file_paths[i_sample])
                assert os.path.exists(
                    save_fp
                ), f"{save_fp} does not exist! Please check the `input_latents_dir` or check if you run `--save_latents` ahead."
                loaded_latents.append(np.load(save_fp))
            loaded_latents = np.stack(loaded_latents) if loaded_latents[0].ndim == 4 else np.concatenate(loaded_latents)
            decode_data = vae.decode(ms.Tensor(loaded_latents) / args.sd_scale_factor)
            decode_data = ms.ops.clip_by_value(
                (decode_data + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0
            ).asnumpy()
            for i_sample in range(args.batch_size):
                save_fp = os.path.join(save_dir, file_paths[i_sample]).replace(".npy", ".gif")
                save_video_data = decode_data[i_sample : i_sample + 1].transpose(
                    0, 2, 3, 4, 1
                )  # (b c t h w) -> (b t h w c)
                save_videos(save_video_data, save_fp, loop=0)
        sys.exit()

    assert args.condition == "text", "LatteT2V only support text condition"
    assert args.text_encoder == "t5", "LatteT2V only support t5 text encoder"
    logger.info("T5 init")
    text_encoder = T5Embedder(
        cache_dir=args.t5_cache_folder,
        pretrained_ckpt=os.path.join(args.t5_cache_folder, "model.ckpt"),
        model_max_length=args.t5_max_length,
    )
    tokenizer = text_encoder.tokenizer

    # 3. build inference pipeline
    scheduler = DDIMScheduler() if args.ddim_sampling else DDPMScheduler()
    text_encoder = text_encoder.model
    pipeline = VideoGenPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        transformer=latte_model,
        vae_scale_factor=args.sd_scale_factor,
    )

    # 4. print key info
    num_params_vae, num_params_vae_trainable = count_params(vae)
    num_params_latte, num_params_latte_trainable = count_params(latte_model)
    num_params = num_params_vae + num_params_latte
    num_params_trainable = num_params_vae_trainable + num_params_latte_trainable
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"Num of samples: {n}",
            f"Num params: {num_params:,} (latte: {num_params_latte:,}, vae: {num_params_vae:,})",
            f"Num trainable params: {num_params_trainable:,}",
            f"Use model dtype: {model_dtype}",
            f"Sampling steps {args.sampling_steps}",
            f"DDIM sampling: {args.ddim_sampling}",
            f"CFG guidance scale: {args.guidance_scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    logger.info(f"Sampling for {n} samples with condition {args.condition}")
    start_time = time.time()

    # infer
    for step, data in tqdm(enumerate(ds_iter), total=dataset_size):
        prompt = [x for x in data["caption"]]
        file_paths = data["file_path"]
        videos = pipeline(
            prompt,
            video_length=video_length,
            height=args.image_size,
            width=args.image_size,
            num_inference_steps=args.sampling_steps,
            guidance_scale=args.guidance_scale,
            enable_temporal_attentions=True,
            mask_feature=False,
            output_type="latents" if args.save_latents else "pil",
        ).video.asnumpy()
        for i_sample in range(args.batch_size):
            file_path = os.path.join(save_dir, file_paths[i_sample])
            if args.save_latents:
                assert ".npy" in file_path, "Only support .npy for saving latent data"
                np.save(file_path, videos[i_sample : i + i_sample])
            else:
                assert ".gif" in file_path or ".mp4" in file_path, "Only support .gif or .mp4 for saving video data"
                videos = videos[i_sample : i + i_sample].transpose(0, 2, 3, 4, 1)  # (b c t h w) -> (b t h w c)
                save_videos(save_video_data, file_path, loop=0)
    end_time = time.time()
    time_cost = end_time - start_time
    logger.info(f"Inference time cost: {time_cost:0.3f}s")
    logger.info(f"Inference speed: {n / time_cost:0.3f} samples/s")
    logger.info(f"{'latents' if args.save_latents else 'videos' } saved to {save_dir}")
