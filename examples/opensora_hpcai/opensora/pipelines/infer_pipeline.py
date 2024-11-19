from typing import Any, Dict, Optional, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7

import logging

import numpy as np

import mindspore as ms
from mindspore import Tensor, ops

from mindone.models.modules.pos_embed import get_2d_sincos_pos_embed
from mindone.transformers import T5EncoderModel

from ..models.layers.rotary_embedding import get_3d_rotary_pos_embed, precompute_freqs_cis
from ..models.vae import AutoencoderKLCogVideoX
from ..models.vae.vae import VideoAutoencoderKL, VideoAutoencoderPipeline
from ..schedulers.iddpm import create_diffusion
from ..schedulers.rectified_flow import RFLOW

__all__ = ["InferPipeline", "InferPipelineFiTLike", "InferPipelineCogVideoX"]

_logger = logging.getLogger()


class InferPipeline:
    """An Inference pipeline for diffusion model

    Args:
        model (nn.Cell): A noise prediction model to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images or videos to and from latent representations.
        scale_factor (float): scale_factor for vae.
        guidance_rescale (float): A higher guidance scale value for noise rescale.
        num_inference_steps: (int): The number of denoising steps.
        sampling (str): sampling method, should be one of ['ddpm', 'ddim', 'rflow'].
    """

    def __init__(
        self,
        model,
        vae,
        text_encoder=None,
        scale_factor=1.0,
        guidance_rescale=1.0,
        guidance_channels: Optional[int] = None,
        num_inference_steps=50,
        sampling: Literal["ddpm", "ddim", "rflow"] = "ddpm",
        micro_batch_size=None,
        diffusion_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.model = model

        self.vae = vae
        self.micro_batch_size = micro_batch_size
        self.scale_factor = scale_factor
        self.guidance_rescale = guidance_rescale
        self.guidance_channels = guidance_channels
        if self.guidance_rescale > 1.0:
            self.use_cfg = True
        else:
            self.use_cfg = False

        self.text_encoder = text_encoder

        if diffusion_config is None:
            diffusion_config = dict()
        self.diffusion = create_diffusion(str(num_inference_steps), **diffusion_config)

        if sampling.lower() == "ddim":
            self.sampling_func = self.diffusion.ddim_sample_loop
        elif sampling.lower() == "ddpm":
            self.sampling_func = self.diffusion.p_sample_loop
        elif sampling.lower() == "rflow":
            self.sampling_func = RFLOW(num_inference_steps, cfg_scale=guidance_rescale, use_timestep_transform=True)
        else:
            raise ValueError(f"Unknown sampling method {sampling}")

    # @ms.jit
    def vae_encode(self, x: Tensor) -> Tensor:
        """
        Image encoding with spatial vae
        Args:
            x: (b c t h w), image (t=1) or video
        """
        return self.vae.encode(x)

    def vae_decode(self, x: Tensor) -> Tensor:
        """
        Image decoding with spatial vae
        Args:
            x: (b c h w), denoised latent
        Return:
            y: (b H W 3), batch of images, normalized to [0, 1]
        """
        if isinstance(self.vae, VideoAutoencoderKL):
            spatial_vae = self.vae
        elif isinstance(self.vae, VideoAutoencoderPipeline):
            spatial_vae = self.vae.spatial_vae

        y = ops.stop_gradient(spatial_vae.module.decode(x) / spatial_vae.scale_factor)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

        # (b 3 H W) -> (b H W 3)
        y = ops.transpose(y, (0, 2, 3, 1))

        return y

    def vae_decode_video(self, x, num_frames=None):
        """
        Args:
            x: (b c t h w), denoised latent
        Return:
            y: (b f H W 3), batch of images, normalized to [0, 1]
        """

        y = ops.stop_gradient(self.vae.decode(x, num_frames=num_frames))
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)
        # (b 3 t h w) -> (b t h w 3)
        y = ops.transpose(y, (0, 2, 3, 4, 1))
        return y

    def data_prepare(self, inputs):
        x = inputs["noise"]
        mask = inputs.get("mask", None)

        if inputs["text_emb"] is None:
            text_tokens = inputs["text_tokens"]
            text_emb = self.get_condition_embeddings(text_tokens, **{"mask": mask}).to(ms.float32)
        else:
            text_emb = inputs["text_emb"]
            b, max_tokens, d = text_emb.shape

        # torch use y_embedding genearted during stdit training,
        # for token/text drop in caption embedder for condition-free guidance training. The null mask is the same as text mask.
        n = x.shape[0]
        # (n_tokens, dim_emb) -> (b n_tokens dim_emb)
        null_emb = self.model.y_embedder.y_embedding[None, :, :].repeat(n, axis=0)

        if self.use_cfg:
            y = ops.cat([text_emb, null_emb], axis=0)
            x_in = ops.concat([x] * 2, axis=0)
            assert y.shape[0] == x_in.shape[0], "shape mismatch!"
            inputs["mask"] = ops.concat([mask, mask], axis=0)
        else:
            x_in = x
            y = text_emb

        # to match stdit input format
        y = ops.expand_dims(y, axis=1)
        return x_in, y

    def get_condition_embeddings(self, text_tokens, **kwargs):
        # text conditions inputs for cross-attention
        if isinstance(self.text_encoder, T5EncoderModel):
            text_emb = ops.stop_gradient(self.text_encoder(text_tokens, attention_mask=kwargs.get("mask", None)))[0]
        else:
            text_emb = ops.stop_gradient(self.text_encoder(text_tokens, **kwargs))

        return text_emb

    def __call__(
        self,
        inputs: dict,
        frames_mask: Optional[Tensor] = None,
        num_frames: int = None,
        additional_kwargs: Optional[dict] = None,
    ) -> Tuple[Union[Tensor, None], Tensor]:
        """
        args:
            inputs: dict

        return:
            images (b H W 3)
        """
        z, y = self.data_prepare(inputs)

        mask = inputs.get("mask", None)
        model_kwargs = dict(y=y)
        if mask is not None:
            model_kwargs["mask"] = mask

        if additional_kwargs is not None:
            model_kwargs.update(additional_kwargs)

        if self.use_cfg:
            model_kwargs.update({"cfg_scale": self.guidance_rescale, "cfg_channel": self.guidance_channels})
            latents = self.sampling_func(
                self.model.construct_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                frames_mask=frames_mask,
            )
            latents, _ = latents.chunk(2, axis=0)
        else:
            latents = self.sampling_func(
                self.model,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                frames_mask=frames_mask,
            )

        if self.vae is not None:
            # latents: (b c t h w)
            # out: (b T H W C)
            images = self.vae_decode_video(latents, num_frames=num_frames)
            return images, latents
        else:
            return None, latents


class InferPipelineFiTLike(InferPipeline):
    def __init__(
        self,
        *args,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        max_image_size: int = 512,
        max_num_frames: int = 16,
        embed_dim: int = 1152,
        num_heads: int = 16,
        vae_downsample_rate: float = 8.0,
        in_channels: int = 4,
        input_sq_size: int = 512,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.p = patch_size
        self.c = in_channels

        self.vae_downsample_rate = vae_downsample_rate
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.input_sq_size = input_sq_size

        self.max_size = int(max_image_size / self.vae_downsample_rate)
        self.max_num_frames = max_num_frames

        self.max_length = int(self.max_size**2 / self.p[1] / self.p[2])

    def _patchify(self, x: Tensor) -> Tensor:
        # N, C, F, H, W -> N, F, T, D
        assert self.p[0] == 1
        n, _, f, h, w = x.shape
        nh, nw = h // self.p[1], w // self.p[2]
        x = ops.reshape(x, (n, self.c, f, nh, self.p[1], nw, self.p[2]))
        x = ops.transpose(x, (0, 2, 3, 5, 1, 4, 6))
        x = ops.reshape(x, (n, f, nh * nw, self.c * self.p[1] * self.p[2]))
        return x

    def _unpatchify(self, x: Tensor, nh: int, nw: int) -> Tensor:
        # N, F, T, D -> N, C, F, H, W
        assert self.p[0] == 1
        n, f, _, _ = x.shape
        x = ops.reshape(x, (n, f, nh, nw, self.c, self.p[1], self.p[2]))
        x = ops.transpose(x, (0, 4, 1, 2, 5, 3, 6))
        x = ops.reshape(x, (n, self.c, f, nh * self.p[1], nw * self.p[2]))
        return x

    def _pad_latent(self, x: Tensor) -> Tensor:
        # N, C, F, H, W -> N, C, max_num_frames, max_size, max_size
        n, c, f, h, w = x.shape
        nh, nw = self.max_size // self.p[1], self.max_size // self.p[2]

        x_fill = self._patchify(x)
        if x_fill.shape[1] > self.max_num_frames and x_fill.shape[2] > self.max_length:
            return x
        elif x_fill.shape[1] <= self.max_num_frames and x_fill.shape[2] > self.max_length:
            x_ = ops.zeros((n, c, self.max_num_frames, h, w), dtype=x.dtype)
            x_[:, :, : x.shape[2]] = x
            return x_

        x = ops.zeros(
            (n, max(f, self.max_num_frames), max(x_fill.shape[2], self.max_length), x_fill.shape[3]), dtype=x.dtype
        )
        x[:, : x_fill.shape[1], : x_fill.shape[2]] = x_fill
        x = self._unpatchify(x, nh, nw)
        return x

    def _unpad_latent(self, x: Tensor, valid_f: int, valid_t: int, h: int, w: int) -> Tensor:
        # N, C, max_num_frames, max_size, max_size -> N, C, F, H, W
        nh, nw = h // self.p[1], w // self.p[2]
        x = self._patchify(x)
        x = x[:, :valid_f, :valid_t]
        x = self._unpatchify(x, nh, nw)
        return x

    def _get_dynamic_size(self, h: int, w: int) -> Tuple[int, int]:
        if h % self.p[1] != 0:
            h += self.p[1] - h % self.p[1]
        if w % self.p[2] != 0:
            w += self.p[2] - w % self.p[2]

        h = h // self.p[1]
        w = w // self.p[2]
        return h, w

    def _create_spatial_pos_embed(self, h: int, w: int) -> Tuple[Tensor, int]:
        """1, T, D"""
        rs = (h * w * self.vae_downsample_rate**2) ** 0.5
        ph, pw = self._get_dynamic_size(h, w)
        scale = rs / self.input_sq_size
        base_size = round((ph * pw) ** 0.5)

        nh, nw = h // self.p[1], w // self.p[2]
        pos_embed_fill = get_2d_sincos_pos_embed(self.embed_dim, nh, nw, scale=scale, base_size=base_size)

        if pos_embed_fill.shape[0] > self.max_length:
            pos_embed = pos_embed_fill
        else:
            pos_embed = np.zeros((self.max_length, self.embed_dim), dtype=np.float32)
            pos_embed[: pos_embed_fill.shape[0]] = pos_embed_fill

        pos_embed = pos_embed[None, ...]
        pos_embed = Tensor(pos_embed, dtype=ms.float32)
        return pos_embed, pos_embed_fill.shape[0]

    def _create_temporal_pos_embed(self, f: int) -> Tuple[Tensor, int]:
        """1, T, D"""
        pos_embed_fill = precompute_freqs_cis(f, self.embed_dim // self.num_heads)

        if pos_embed_fill.shape[0] > self.max_num_frames:
            pos_embed = pos_embed_fill
        else:
            pos_embed = np.zeros((self.max_num_frames, self.embed_dim // self.num_heads), dtype=np.float32)
            pos_embed[: pos_embed_fill.shape[0]] = pos_embed_fill

        pos_embed = pos_embed[None, ...]
        pos_embed = Tensor(pos_embed, dtype=ms.float32)
        return pos_embed, pos_embed_fill.shape[0]

    def _create_mask(self, t: int, valid_t: int, n: int) -> Tensor:
        """N, T"""
        if valid_t > t:
            mask = np.ones((valid_t,), dtype=np.bool_)
        else:
            mask = np.zeros((t,), dtype=np.bool_)
            mask[:valid_t] = True
        mask = np.tile(mask[None, ...], (n, 1))
        mask = Tensor(mask, dtype=ms.uint8)
        return mask

    def __call__(
        self,
        inputs: dict,
        frames_mask: Optional[Tensor] = None,
        additional_kwargs: Optional[dict] = None,
    ) -> Tuple[Union[Tensor, None], Tensor]:
        """
        args:
            inputs: dict

        return:
            images (b H W 3)
        """
        z, y = self.data_prepare(inputs)
        n, c, f, h, w = z.shape
        assert self.c == c

        z = self._pad_latent(z)
        spatial_pos, valid_t = self._create_spatial_pos_embed(h, w)
        temporal_pos, valid_f = self._create_temporal_pos_embed(f)
        spatial_mask = self._create_mask(self.max_length, valid_t, n)
        temporal_mask = self._create_mask(self.max_num_frames, valid_f, n)

        mask = inputs.get("mask", None)
        model_kwargs = dict(y=y)
        if mask is not None:
            model_kwargs["mask"] = mask

        if additional_kwargs is not None:
            model_kwargs.update(additional_kwargs)

        model_kwargs["spatial_pos"] = spatial_pos
        model_kwargs["spatial_mask"] = spatial_mask
        model_kwargs["temporal_pos"] = temporal_pos
        model_kwargs["temporal_mask"] = temporal_mask

        # HACK: to make the frame_mask valid with padded length
        frames_mask_ = temporal_mask[0:1]
        frames_mask_[:, : frames_mask.shape[1]] = frames_mask

        if self.use_cfg:
            model_kwargs.update({"cfg_scale": self.guidance_rescale, "cfg_channel": self.guidance_channels})
            latents = self.sampling_func(
                self.model.construct_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                frames_mask=frames_mask_,
            )
            latents, _ = latents.chunk(2, axis=0)
        else:
            latents = self.sampling_func(
                self.model,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                frames_mask=frames_mask_,
            )

        latents = self._unpad_latent(latents, valid_f, valid_t, h, w)

        if self.vae is not None:
            if latents.dim() == 4:
                images = self.vae_decode(latents)
            else:
                # latents: (b c t h w)
                # out: (b T H W C)
                images = self.vae_decode_video(latents)
            return images, latents
        else:
            return None, latents


class InferPipelineCogVideoX(InferPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.vae, AutoencoderKLCogVideoX)
        self.vae.enable_slicing()
        self.vae.enable_tiling()

    def vae_decode_video(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (b t c h w), denoised latent
        Return:
            y: (b f H W 3), batch of images, normalized to [0, 1]
        """
        y = ops.stop_gradient(self.vae.decode(x.to(self.vae.dtype)) / self.vae.scaling_factor)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)
        # (b 3 t h w) -> (b t h w 3)
        y = ops.transpose(y, (0, 2, 3, 4, 1))
        return y

    def data_prepare(self, inputs):
        x = inputs["noise"]

        if inputs["text_emb"] is None:
            text_tokens = inputs["text_tokens"]
            text_emb = self.get_condition_embeddings(text_tokens).to(ms.float32)
        else:
            text_emb = inputs["text_emb"]

        if self.use_cfg:
            if inputs.get("null_text_emb", None) is None:
                null_text_tokens = ops.broadcast_to(inputs["null_text_tokens"], text_tokens.shape)
                null_text_emb = self.get_condition_embeddings(null_text_tokens).to(ms.float32)
            else:
                null_text_emb = inputs["null_text_emb"]

            y = ops.cat([text_emb, null_text_emb], axis=0)
            x_in = ops.concat([x] * 2, axis=0)
            assert y.shape[0] == x_in.shape[0], "shape mismatch!"
        else:
            x_in = x
            y = text_emb

        return x_in, y

    def get_resize_crop_region_for_grid(self, src, tgt_width, tgt_height):
        tw = tgt_width
        th = tgt_height
        h, w = src
        r = h / w
        if r > (th / tw):
            resize_height = th
            resize_width = int(round(th / h * w))
        else:
            resize_width = tw
            resize_height = int(round(tw / w * h))

        crop_top = int(round((th - resize_height) / 2.0))
        crop_left = int(round((tw - resize_width) / 2.0))
        return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

    def prepare_rotary_positional_embeddings(self, height: int, width: int, num_frames: int) -> np.ndarray:
        grid_height = height // self.model.patch_size[1]
        grid_width = width // self.model.patch_size[2]
        base_size_width = self.model.sample_width // self.model.patch_size[2]
        base_size_height = self.model.sample_height // self.model.patch_size[1]
        base_num_frames = (num_frames + self.model.patch_size[0] - 1) // self.model.patch_size[0]

        grid_crops_coords = self.get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.model.hidden_size // self.model.num_heads,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type=self.model.rope_grid_type,
            max_size=(base_size_height, base_size_width),
        )

        return np.stack([freqs_cos, freqs_sin])[None]

    def __call__(
        self,
        inputs: dict,
        additional_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[Union[Tensor, None], Tensor]:
        z, y = self.data_prepare(inputs)

        if self.model.use_rotary_positional_embeddings:
            image_rotary_emb = self.prepare_rotary_positional_embeddings(z.shape[-2], z.shape[-1], z.shape[-3])
            image_rotary_emb = Tensor(np.tile(image_rotary_emb, (z.shape[0], 1, 1, 1)))
        else:
            image_rotary_emb = None

        model_kwargs = dict(y=y, image_rotary_emb=image_rotary_emb)

        if additional_kwargs is not None:
            model_kwargs.update(additional_kwargs)

        if self.use_cfg:
            model_kwargs.update({"cfg_scale": self.guidance_rescale, "cfg_channel": self.guidance_channels})
            latents = self.sampling_func(
                self.model.construct_with_cfg,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
            )
            latents, _ = latents.chunk(2, axis=0)
        else:
            latents = self.sampling_func(
                self.model,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
            )

        if self.vae is not None:
            # latents: (b c t h w)
            # out: (b T H W C)
            try:
                images = self.vae_decode_video(latents)
                return images, latents
            except Exception as e:
                _logger.warning(f"Cannot decode video. {repr(e)}.")
                return None, latents
        else:
            return None, latents
