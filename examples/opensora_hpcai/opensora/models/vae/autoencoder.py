import sys

import logging
import os

import numpy as np
from .autoencoder_kl import AutoencoderKL as AutoencoderKL_SD

import mindspore as ms
from mindspore import nn, ops

__all__ = ["AutoencoderKL"]


_logger = logging.getLogger(__name__)
SD_CONFIG = {
    "double_z": True,
    "z_channels": 4,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0,
}


class AutoencoderKL(AutoencoderKL_SD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def init_from_ckpt(self, path, ignore_keys=list()):
        if not os.path.exists(path):
            raise ValueError(
                "Maybe download failed. Please download the VAE encoder from https://huggingface.co/stabilityai/sd-vae-ft-mse"
            )
        param_dict = ms.load_checkpoint(path)
        param_not_load, ckpt_not_load = ms.load_param_into_net(self, param_dict, strict_load=True)
        if param_not_load or ckpt_not_load:
            _logger.warning(
                f"{param_not_load} in network is not loaded or {ckpt_not_load} in checkpoint is not loaded!"
            )
