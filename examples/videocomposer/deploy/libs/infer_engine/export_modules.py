import mindspore as ms
from mindspore import nn, ops


class DataPrepare(nn.Cell):
    """
    Some data prepare process. like text encode, image encode.

    Args:
        text_encoder(nn.Cell): Frozen text-encoder.
        vae(nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scheduler(nn.Cell): A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        scale_factor(float): scale_factor for vae
    """

    def __init__(self, text_encoder, vae, scheduler, scale_factor=1.0, clip_image_encoder=None, extra_conds=None, frames=16):
        super(DataPrepare, self).__init__()
        self.text_encoder = text_encoder
        self.vae = vae
        self.scheduler = scheduler
        self.scale_factor = scale_factor
        self.alphas_cumprod = scheduler.alphas_cumprod

        self.clip_image_encoder = clip_image_encoder
        self.extra_conds = extra_conds
        self.frames = frames

    def vae_encode(self, x):
        # 
        image_latents = self.vae.encode(x)
        image_latents = image_latents * self.scale_factor
        return image_latents.astype(ms.float16)

    def latents_add_noise(self, image_latents, noise, ts):
        latents = self.scheduler.add_noise(image_latents, noise, self.alphas_cumprod[ts])
        return latents

    def prompt_embed(self, prompt_data, negative_prompt_data):
        # inputs: [bs, 77], [bs, 77]
        # outputs: [2*bs, 77, 1024] 
        pos_prompt_embeds = self.text_encoder(prompt_data)
        # TODO: originally, if  use_fps_condition is True, negative_prompt_embeds is zero-like vectors 
        negative_prompt_embeds = self.text_encoder(negative_prompt_data)
        prompt_embeds = ops.concat([negative_prompt_embeds, pos_prompt_embeds], axis=0)

        return prompt_embeds

    def style_embed(self, style_image):
        # input: [bs, 3, 224, 224]
        # outut: [bs, 1, 1024] 
        style_emb = self.clip_image_encoder(style_image)
        style_emb = ops.unsqueeze(style_emb, 1) # TODO: need to convert to fp16?

        return style_emb

    def single_image_transform(self, single_image): 
        # input: (bs 1 c 384 384)
        # output: (bs c f 384 384)
        single_image = ops.tile(single_image, (1, self.frames, 1, 1, 1))
        single_image = ops.transpose(single_image, (0, 2, 1, 3, 4))
        return single_image
    
    def motion_transform(self, motion_vectors):
        # (bs f 2 256 256) ->  (bs 2 f 256 256)
        motion_vectors = ops.transpose(motion_vectors, (0, 2, 1, 3, 4))
        return motion_vectors

    def fps_transform(self, fps):
        # TODO: like noise, may be we need to some usell op to make sure fps will not be eliminate
        return fps


class MotionStyleTransferDataPrepare(DataPrepare):
    def __init__(self, text_encoder, vae, scheduler, scale_factor=1.0, clip_image_encoder=None, extra_conds=None, frames=16):
        super(MotionStyleTransferDataPrepare, self).__init__(text_encoder, vae, scheduler, scale_factor, clip_image_encoder, extra_conds, frames)

    def construct(self, prompt_data, negative_prompt_data, noise, style_image, single_image, motion_vectors, fps):
        '''
        fps: frame rate, shape (1,), ms.Tensor int
        noise: (b z f h//8 w//8), z=4 
        '''
        text_emb = self.prompt_embed(prompt_data, negative_prompt_data)
        style_emb = self.style_embed(style_image)
        single_image_tr = self.single_image_transform(single_image) 
        motion_vectors_tr = self.motion_transform(motion_vectors)

        return text_emb, style_emb, single_image_tr, motion_vectors_tr, fps, noise


class MotionStyleTransferPredictNoise(nn.Cell):
    """
    Predict the noise residual.

    Args:
        unet (nn.Cell): A `UNet2DConditionModel` to denoise the encoded image latents.
        guidance_rescale (float): A higher guidance scale value for noise rescale.
    """

    def __init__(self, unet, guidance_rescale=0.0):
        super(MotionStyleTransferPredictNoise, self).__init__()
        self.unet = unet
        self.guidance_rescale = guidance_rescale

    def predict_noise(self, x, t_continuous, text_emb, style_emb, single_image, motion_vectors, fps, guidance_scale):
        """
        x: noise (b z f h//8 w//8), z=4 
        t_continous:  ms.Tensor, int
        text_emb: (bs*2 77 1024) for pos and neg prompt
        style_emb: (bs 1 1024)
        single_image: (bs 3 f 384 384)
        motion_vectors: (bs 2 f 256 256)
        fps:  ms.Tensor, int
        guidance_scale (ms.Tensor float): A higher guidance scale value encourages the model to generate images closely linked to the text
            prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
        """
        
        t_continuous = ops.tile(t_continuous.reshape(1), (x.shape[0],)) # (bs, )
        t_in = ops.concat([t_continuous] * 2, axis=0) # (bs*2, )
        fps = ops.tile(fps.reshape(1), (x.shape[0],))
        fps = ops.concat([fps] * 2, axis=0) # TODO: not used by default. Check effectivness if used.

        x_in = ops.concat([x] * 2, axis=0)
        style_emb = ops.concat([style_emb] * 2, axis=0)
        single_image = ops.concat([single_image] * 2, axis=0)
        motion_vectors = ops.concat([motion_vectors] * 2, axis=0)
        
        print("D--: x_in ", x_in.shape)
        print("D--: t_in ", t_in.shape)
        print("D--: text_emb", text_emb.shape)
        print("D--: style_emb", style_emb.shape)
        noise_pred = self.unet(
                    x_in,
                    t_in,
                    y=text_emb,
                    image=style_emb,
                    local_image=single_image,
                    motion=motion_vectors,
                    fps=fps,
                    #depth=depth,
                    #sketch=sketch,
                ) # (bs*2 4 f h//8 w//8)

        noise_pred_uncond, noise_pred_text = ops.split(noise_pred, split_size_or_sections=noise_pred.shape[0] // 2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        if self.guidance_rescale > 0:
            noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text)
        return noise_pred

    def rescale_noise_cfg(self, noise_pred, noise_pred_text):
        """
        Rescale `noise_pred` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = ops.std(noise_pred_text, axis=tuple(range(1, len(noise_pred_text.shape))), keepdims=True)
        std_cfg = ops.std(noise_pred, axis=tuple(range(1, len(noise_pred.shape))), keepdims=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_pred * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_pred = self.guidance_rescale * noise_pred_rescaled + (1 - self.guidance_rescale) * noise_pred
        return noise_pred

    def construct(self, latents, ts, text_emb, style_emb, single_image, motion_vectors, fps, guidance_scale):
        return self.predict_noise(latents, ts, text_emb, style_emb, single_image, motion_vectors, fps, guidance_scale)


class SchedulerPreProcess(nn.Cell):
    """
    Pre process of sampler.

    Args:
        scheduler (nn.Cell): A scheduler to be used in combination with `unet` to denoise the encoded image latents.
    """

    def __init__(self, scheduler):
        super(SchedulerPreProcess, self).__init__()
        self.scheduler = scheduler

    def construct(self, latents, t):
        return self.scheduler.scale_model_input(latents, t)


class NoisySample(nn.Cell):
    """
    Compute the previous noisy sample x_t -> x_t-1.

    Args:
        scheduler (nn.Cell): A scheduler to be used in combination with `unet` to denoise the encoded image latents.
    """

    def __init__(self, scheduler):
        super(NoisySample, self).__init__()
        self.scheduler = scheduler

    def construct(self, noise_pred, ts, latents, num_inference_steps):
        return self.scheduler(noise_pred, ts, latents, num_inference_steps)


class VAEDecoder(nn.Cell):
    """
    VAE Decoder

    Args:
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scale_factor (float): scale_factor for vae.
    """

    def __init__(self, vae, scale_factor=1.0):
        super(VAEDecoder, self).__init__()
        self.vae = vae
        self.scale_factor = scale_factor

    def vae_decode(self, x):
        '''
        x: latent frames (b c f h w) = (bs, 4, f, H// 8, W// 8) 
        returns: (bs f C H W) = (bs f 3 H W)
        '''
        bs, c, f, h, w = x.shape

        # (b c f h w) -> (b f c h w) -> (b*f c h w)
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        x = ops.reshape(x, (-1, x.shape[2], x.shape[3], x.shape[4]))
        
        # vae decode: (b*f 4 H//8 W//8) => (b*f 3 H W)
        y = self.vae.decode(x / self.scale_factor)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

        # (b*f 3 H W) -> (b f 3 H W)
        y = ops.reshape(y, (bs, y.shape[0]//bs, y.shape[1], y.shape[2], y.shape[3]))

        return y

    def construct(self, latents):
        return self.vae_decode(latents)
