from abc import ABC

from ..diffusion import create_diffusion

import mindspore as ms
from mindspore import ops

__all__ = ["InferPipeline"]


class InferPipeline(ABC):
    """An Inference pipeline for diffusion model

    Args:
        model (nn.Cell): A noise prediction model to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scale_factor (float): scale_factor for vae.
        guidance_rescale (float): A higher guidance scale value for noise rescale.
        num_inference_steps: (int): The number of denoising steps.
        ddim_sampling: (bool): whether to use DDIM sampling. If False, will use DDPM sampling.
    """

    def __init__(
        self,
        model,
        vae,
        text_encoder=None,
        condition: str = None,
        scale_factor=1.0,
        guidance_rescale=1.0,
        num_inference_steps=50,
        ddim_sampling=True,
    ):
        super().__init__()
        self.model = model
        self.condition = condition

        self.vae = vae
        self.scale_factor = scale_factor
        self.guidance_rescale = guidance_rescale
        if self.guidance_rescale > 1.0:
            self.use_cfg = True
        else:
            self.use_cfg = False

        self.text_encoder = text_encoder
        self.diffusion = create_diffusion(str(num_inference_steps))
        if ddim_sampling:
            self.sampling_func = self.diffusion.ddim_sample_loop
        else:
            self.sampling_func = self.diffusion.p_sample_loop

    @ms.jit
    def vae_encode(self, x):
        image_latents = self.vae.encode(x)
        image_latents = image_latents * self.scale_factor
        return image_latents.astype(ms.float16)

    @ms.jit
    def vae_decode(self, x):
        """
        Args:
            x: (b c h w), denoised latent
        Return:
            y: (b H W 3), batch of images, normalized to [0, 1]
        """
        b, c, h, w = x.shape

        y = self.vae.decode(x / self.scale_factor)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

        # (b 3 H W) -> (b H W 3)
        y = ops.transpose(y, (0, 2, 3, 1))

        return y

    def vae_decode_video(self, x):
        """
        Args:
            x: (b c t h w), denoised latent
        Return:
            y: (b f H W 3), batch of images, normalized to [0, 1]
        """
        y = []
        for x_sample in x:
            # c t h w -> t c h w
            x_sample = x_sample.permute(1, 0, 2, 3)
            y.append(self.vae_decode(x_sample))
        y = ops.stack(y, axis=0)

        return y

    def data_prepare(self, inputs):
        x = inputs["noise"]

        if self.condition == "text":
            text_tokens = inputs["text_tokens"]
            mask = inputs.get("mask", None)
            text_emb = self.get_condition_embeddings(text_tokens, **{"mask": mask})
            if self.use_cfg:
                y, y_null = text_emb, ops.zeros_like(text_emb)
                y = ops.cat([y, y_null], axis=0)
                x_in = ops.concat([x] * 2, axis=0)
                assert y.shape[0] == x_in.shape[0], "shape mismatch!"
                if mask is not None:
                    inputs["mask"] = ops.concat([mask] * 2, axis=0)
            else:
                x_in = x
                y = text_emb
        else:
            if self.use_cfg:
                y = ops.cat([inputs["y"], inputs["y_null"]], axis=0)
                x_in = ops.concat([x] * 2, axis=0)
                assert y.shape[0] == x_in.shape[0], "shape mismatch!"
            else:
                x_in = x
                y = inputs["y"]
        return x_in, y

    def get_condition_embeddings(self, text_tokens, **kwargs):
        # text conditions inputs for cross-attention
        text_emb = ops.stop_gradient(self.text_encoder(text_tokens, **kwargs))
        return text_emb

    def __call__(self, inputs):
        """
        args:
            inputs: dict

        return:
            images (b H W 3)
        """
        z, y = self.data_prepare(inputs)
        # prepare model key arguments for sampling
        # class condition, keys include "y", "cfg_scale"(optional)
        # None condition, keys include "y", "cfg_scale"(optional)
        # text condition, keys include "text_embed", "mask"(optional), "cfg_scale"(optional)
        if self.condition == "text":
            mask = inputs.get("mask", None)
            model_kwargs = dict(text_embed=y)
            if mask is not None:
                model_kwargs["mask"] = mask
        else:
            model_kwargs = dict(y=y)

        if self.use_cfg:
            model_kwargs["cfg_scale"] = self.guidance_rescale
            latents = self.sampling_func(
                self.model.construct_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True
            )
            latents, _ = latents.chunk(2, axis=0)
        else:
            latents = self.sampling_func(
                self.model.construct, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True
            )
        if latents.dim() == 4:
            images = self.vae_decode(latents)
        else:
            # latents: (b c t h w)
            # out: (b T H W C)
            images = self.vae_decode_video(latents)
        return images
