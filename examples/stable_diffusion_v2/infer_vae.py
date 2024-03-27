import argparse
import logging
import os
import sys
import time
from math import log10, sqrt

import numpy as np
from ldm.data.dataset_vae import ImageDataset, create_dataloader
from ldm.models.autoencoder import GeneratorWithLoss
from ldm.models.lpips import LPIPS
from ldm.util import instantiate_from_config, str2bool
from omegaconf import OmegaConf
from PIL import Image, ImageSequence
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from tqdm import tqdm
from utils import model_utils

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
from mindone.utils.logger import set_logger

logger = logging.getLogger(__name__)


def load_model_weights(model, ckpt, verbose=True, prefix_filter=["first_stage_model.", "autoencoder."]):
    sd = ms.load_checkpoint(ckpt)

    # filter out vae weights and rename
    all_sd_pnames = list(sd.keys())
    for pname in all_sd_pnames:
        is_vae_param = False
        for pf in prefix_filter:
            if pname.startswith(pf):
                sd[pname.replace(pf, "")] = sd.pop(pname)
                is_vae_param = True
        if not is_vae_param:
            sd.pop(pname)
    vae_state_dict = sd
    # logger.info(list(sd.keys()))
    m, u = model_utils.load_param_into_net_with_filter(model, vae_state_dict)

    if len(m) > 0 and verbose:
        logger.info("missing keys:")
        logger.info(m)
    if len(u) > 0 and verbose:
        logger.info("unexpected keys:")
        logger.info(u)

    model.set_train(False)

    return model


def postprocess(x, trim=True):
    pixels = (x + 1) * 127.5
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    # b, c, h, w -> b h w c
    return np.transpose(pixels, (0, 2, 3, 1))


def visualize(recons, x=None, save_fn="tmp_vae_recons"):
    # x: (b h w c), np array
    for i in range(recons.shape[0]):
        if x is not None:
            out = np.concatenate((x[i], recons[i]), axis=-2)
        else:
            out = recons[i]
        Image.fromarray(out).save(f"{save_fn}-{i:02d}.png")


def measure_psnr(original, compressed, return_mean=False):
    # input: (h w c) or (b h w c) in pixel range 0-255
    if len(original.shape) == 3:
        original = np.expand_dims(original, axis=0)
        compressed = np.expand_dims(compressed, axis=0)

    sample_psnr = []
    for i in range(original.shape[0]):
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:  # MSE is zero means no noise is present in the signal .
            # Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        sample_psnr.append(psnr)

    if return_mean:
        return sum(sample_psnr) / len(sample_psnr)
    else:
        return sample_psnr


def infer_vae(args):
    ms.set_context(mode=args.mode)
    set_logger(name="", output_dir=args.output_path, rank=0)

    config = OmegaConf.load(args.model_config)
    model = instantiate_from_config(config.generator)
    model = load_model_weights(model, args.ckpt_path)
    # state_dict = ms.load_checkpoint(ckpt_path, model, specify_prefix=["first_stage_model", "autoencoder"])
    logger.info(f"Loaded checkpoint from  {args.ckpt_path}")

    if args.measure_loss:
        perc_loss_fn = LPIPS()

    model.set_train(False)

    ds_config = dict(
        csv_path=args.csv_path,
        image_folder=args.data_path,
        size=args.size,
        crop_size=args.crop_size,
        flip=False,
        random_crop=False,
    )
    # test source dataset
    dataset = create_dataloader(
        ds_config=ds_config,
        batch_size=args.batch_size,
        num_parallel_workers=args.num_parallel_workers,
        shuffle=False,
        drop_remainder=False,
    )
    dataset_size = dataset.get_dataset_size()

    ds_iter = dataset.create_dict_iterator(1)

    logger.info(f"Inferene begins")
    mean_infer_time = 0
    psnr_res = []
    ssim_res = []
    lpips_res = []
    for step, data in tqdm(enumerate(ds_iter)):
        x = data["image"]
        start_time = time.time()

        z, posterior_mean, posterior_logvar = model.encode(x)
        if not args.encode_only:
            recons = model.decode(z)

        infer_time = time.time() - start_time
        mean_infer_time += infer_time
        logger.info(f"Infer time: {infer_time}")

        if not args.encode_only:
            recons_rgb = postprocess(recons.asnumpy())
            x_rgb = postprocess(x.asnumpy())
            # eval psnr
            psnr_cur = [calc_psnr(x_rgb[i], recons_rgb[i], data_range=255) for i in range(x_rgb.shape[0])]
            ssim_cur = [
                calc_ssim(x_rgb[i], recons_rgb[i], data_range=255, channel_axis=-1) for i in range(x_rgb.shape[0])
            ]
            # print(psnr_cur)
            psnr_res.extend(psnr_cur)
            ssim_res.extend(ssim_cur)
            # logger.info(f"mean psnr:{np.mean(psnr_cur):.4f}")
            # logger.info(f"mean ssim:{np.mean(ssim_cur):.4f}")

            if args.save_images:
                save_fn = os.path.join(
                    args.output_path, "{}-{}".format(os.path.basename(args.data_path), f"step{step:03d}")
                )
                visualize(recons_rgb, x_rgb, save_fn=save_fn)

            if args.measure_loss:
                rec_loss = np.abs((x - recons).asnumpy()).mean()
                perc_loss = perc_loss_fn(x, recons)
                lpips_res.append(perc_loss.asnumpy().mean())

    mean_infer_time /= dataset_size
    logger.info(f"Mean infer time: {mean_infer_time}")
    logger.info(f"Done. Results saved in {args.output_path}")

    if not args.encode_only:
        mean_psnr = sum(psnr_res) / len(psnr_res)
        logger.info(f"mean psnr:{mean_psnr:.4f}")
        logger.info(f"mean ssim:{np.mean(ssim_res):.4f}")
        if args.measure_loss:
            logger.info(f"mean lpips:{np.mean(lpips_res):.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config",
        default="configs/train/autoencoder_kl_f8.yaml",
        type=str,
        help="model architecture config",
    )
    parser.add_argument(
        "--ckpt_path", default="outputs/vae_train/ckpt/vae_kl_f8-e10.ckpt", type=str, help="checkpoint path"
    )
    parser.add_argument("--csv_path", default=None, type=str, help="path to csv annotation file")
    parser.add_argument("--data_path", default="dataset", type=str, help="data path")
    parser.add_argument(
        "--output_path", default="samples/vae_recons", type=str, help="output directory to save inference results"
    )
    parser.add_argument("--size", default=384, type=int, help="image rescale size")
    parser.add_argument("--crop_size", default=256, type=int, help="image crop size")

    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--num_parallel_workers", default=8, type=int, help="num workers for data loading")
    parser.add_argument(
        "--measure_loss",
        default=False,
        type=str2bool,
        help="whether measure loss including reconstruction, kl, perceptual loss",
    )
    parser.add_argument("--save_images", default=True, type=str2bool, help="whether save reconstructed images")
    parser.add_argument("--encode_only", default=False, type=str2bool, help="only encode to save z or distribution")
    parser.add_argument(
        "--save_z_dist",
        default=False,
        type=str2bool,
        help="If True, save z distribution, mean and logvar. Otherwise, save z after sampling.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    infer_vae(args)
