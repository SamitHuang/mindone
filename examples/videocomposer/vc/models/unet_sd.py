import logging
import math
import os

from utils.download import download_checkpoint

import mindspore as ms
import mindspore.common.initializer as init
from mindspore import nn, ops

from ..utils.pt2ms import auto_map
from .attention import (
    BasicTransformerBlock,
    CrossAttention,
    FeedForward,
    GroupNorm,
    RelativePositionBias,
    SiLU,
    SpatialTransformer,
    TemporalAttentionBlock,
    TemporalAttentionMultiBlock,
    TemporalConvBlock_v2,
    TemporalTransformer,
    default,
    is_old_ms_version,
    zero_module,
)
from .droppath import DropPath, DropPathWithControl
from .rotary_embedding import RotaryEmbedding
from .stc_encoder import Transformer_v2

# import time
# from functools import partial


_logger = logging.getLogger(__name__)

__all__ = ["UNetSD_temporal"]

_CKPT_URL = {
    "UNetSD_temporal": "https://download.mindspore.cn/toolkits/mindone/videocomposer/model_weights/non_ema_228000-7f157ec2.ckpt"
}

USE_TEMPORAL_TRANSFORMER = True


# load all keys started with prefix and replace them with new_prefix
def load_Block(state, prefix, new_prefix=None):
    if new_prefix is None:
        new_prefix = prefix

    state_dict = {}

    state = {key: value for key, value in state.items() if prefix in key}

    for key, value in state.items():
        new_key = key.replace(prefix, new_prefix)
        # ### 3DConv
        # if 'encoder.0.weight' in key or 'shortcut.weight' in key or 'layer1.2.weight' in key or 'layer2.3.weight' in key or 'head.2.weight' in key:
        #     state_dict[new_key]=value.unsqueeze(2)
        # else:
        state_dict[new_key] = value

    return state_dict


def load_2d_pretrained_state_dict(state, cfg):
    new_state_dict = {}

    dim = cfg.unet_dim
    num_res_blocks = cfg.unet_res_blocks
    # temporal_attention = cfg.temporal_attention
    # temporal_conv = cfg.temporal_conv
    dim_mult = cfg.unet_dim_mult
    attn_scales = cfg.unet_attn_scales

    # params
    enc_dims = [dim * u for u in [1] + dim_mult]
    dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
    shortcut_dims = []
    scale = 1.0

    # embeddings
    state_dict = load_Block(state, prefix="time_embedding")
    new_state_dict.update(state_dict)
    state_dict = load_Block(state, prefix="y_embedding")
    new_state_dict.update(state_dict)
    state_dict = load_Block(state, prefix="context_embedding")
    new_state_dict.update(state_dict)

    encoder_idx = 0
    # init block
    state_dict = load_Block(state, prefix=f"encoder.{encoder_idx}", new_prefix=f"encoder.{encoder_idx}.0")
    new_state_dict.update(state_dict)
    encoder_idx += 1

    shortcut_dims.append(dim)
    for i, (in_dim, out_dim) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
        for j in range(num_res_blocks):
            # residual (+attention) blocks
            idx = 0
            idx_ = 0
            # residual (+attention) blocks
            state_dict = load_Block(
                state, prefix=f"encoder.{encoder_idx}.{idx}", new_prefix=f"encoder.{encoder_idx}.{idx_}"
            )
            new_state_dict.update(state_dict)
            idx += 1
            idx_ = 2

            if scale in attn_scales:
                # block.append(AttentionBlock(out_dim, context_dim, num_heads, head_dim))
                state_dict = load_Block(
                    state, prefix=f"encoder.{encoder_idx}.{idx}", new_prefix=f"encoder.{encoder_idx}.{idx_}"
                )
                new_state_dict.update(state_dict)
                # if temporal_attention:
                #     block.append(TemporalAttentionBlock(out_dim, num_heads, head_dim, rotary_emb = self.rotary_emb))
            in_dim = out_dim
            encoder_idx += 1
            shortcut_dims.append(out_dim)

            # downsample
            if i != len(dim_mult) - 1 and j == num_res_blocks - 1:
                # downsample = ResidualBlock(out_dim, embed_dim, out_dim, use_scale_shift_norm, 0.5, dropout)
                state_dict = load_Block(state, prefix=f"encoder.{encoder_idx}", new_prefix=f"encoder.{encoder_idx}.0")
                new_state_dict.update(state_dict)

                shortcut_dims.append(out_dim)
                scale /= 2.0
                encoder_idx += 1

    # middle
    middle_idx = 0

    state_dict = load_Block(state, prefix=f"middle.{middle_idx}")
    new_state_dict.update(state_dict)
    middle_idx += 2

    state_dict = load_Block(state, prefix="middle.1", new_prefix=f"middle.{middle_idx}")
    new_state_dict.update(state_dict)
    middle_idx += 1

    for _ in range(cfg.temporal_attn_times):
        # self.middle.append(TemporalAttentionBlock(out_dim, num_heads, head_dim, rotary_emb =  self.rotary_emb))
        middle_idx += 1

    state_dict = load_Block(state, prefix="middle.2", new_prefix=f"middle.{middle_idx}")
    new_state_dict.update(state_dict)
    middle_idx += 2

    decoder_idx = 0
    for i, (in_dim, out_dim) in enumerate(zip(dec_dims[:-1], dec_dims[1:])):
        for j in range(num_res_blocks + 1):
            idx = 0
            idx_ = 0
            # residual (+attention) blocks
            state_dict = load_Block(
                state, prefix=f"decoder.{decoder_idx}.{idx}", new_prefix=f"decoder.{decoder_idx}.{idx_}"
            )
            new_state_dict.update(state_dict)
            idx += 1
            idx_ += 2
            if scale in attn_scales:
                # block.append(AttentionBlock(out_dim, context_dim, num_heads, head_dim))
                state_dict = load_Block(
                    state, prefix=f"decoder.{decoder_idx}.{idx}", new_prefix=f"decoder.{decoder_idx}.{idx_}"
                )
                new_state_dict.update(state_dict)
                idx += 1
                idx_ += 1
                for _ in range(cfg.temporal_attn_times):
                    idx_ += 1

            # in_dim = out_dim

            # upsample
            if i != len(dim_mult) - 1 and j == num_res_blocks:
                # upsample = ResidualBlock(out_dim, embed_dim, out_dim, use_scale_shift_norm, 2.0, dropout)
                state_dict = load_Block(
                    state, prefix=f"decoder.{decoder_idx}.{idx}", new_prefix=f"decoder.{decoder_idx}.{idx_}"
                )
                new_state_dict.update(state_dict)
                idx += 1
                idx_ += 2

                scale *= 2.0
                # block.append(upsample)
            # self.decoder.append(block)
            decoder_idx += 1

    state_dict = load_Block(state, prefix="head")
    new_state_dict.update(state_dict)

    return new_state_dict


def sinusoidal_embedding(timesteps, dim):
    # check input
    half = dim // 2
    timesteps = timesteps.float()

    # compute sinusoidal embedding
    sinusoid = ops.outer(timesteps, ops.pow(10000, -ops.arange(half).to(timesteps.dtype).div(half)))
    x = ops.cat([ops.cos(sinusoid), ops.sin(sinusoid)], axis=1)
    if dim % 2 != 0:
        x = ops.cat([x, ops.zeros_like(x[:, :1])], axis=1)
    return x


def prob_mask_like(shape, prob):
    if prob == 1:
        return ops.ones(shape, dtype=ms.bool_)
    elif prob == 0:
        return ops.zeros(shape, dtype=ms.bool_)
    else:
        mask = ops.uniform(shape, ms.Tensor(0), ms.Tensor(1), dtype=ms.float32) < prob
        # avoid mask all, which will cause find_unused_parameters error
        if mask.all():
            mask[0] = False
        return mask


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class Upsample(nn.Cell):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, dtype=ms.float32):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.dtype = dtype
        if use_conv:
            self.conv = nn.Conv2d(
                self.channels, self.out_channels, 3, pad_mode="pad", padding=padding, has_bias=True
            ).to_float(self.dtype)

    def construct(self, x):
        # assert x.shape[1] == self.channels
        if self.dims == 3:
            # x = ops.interpolate(x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest")
            x = ops.ResizeNearestNeighbor((x.shape[2], x.shape[3] * 2, x.shape[4] * 2))(x)
        else:
            # x = ops.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2), mode="nearest")
            x = ops.ResizeNearestNeighbor((x.shape[2] * 2, x.shape[3] * 2))(x)
        if self.use_conv:
            x = self.conv(x)
        return x


class ResBlock(nn.Cell):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_temporal_conv: if True and out_channels is specified, use
        TemporalConvBlock_v2.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        use_temporal_conv=True,
        use_image_dataset=False,
        dtype=ms.float32,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_temporal_conv = use_temporal_conv
        self.dtype = dtype

        self.in_layers = nn.SequentialCell(
            GroupNorm(32, channels),
            SiLU(),
            nn.Conv2d(channels, self.out_channels, 3, pad_mode="pad", padding=1, has_bias=True).to_float(self.dtype),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, dtype=self.dtype)
            self.x_upd = Upsample(channels, False, dims, dtype=self.dtype)
        elif down:
            self.h_upd = Downsample(channels, False, dims, dtype=self.dtype)
            self.x_upd = Downsample(channels, False, dims, dtype=self.dtype)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.SequentialCell(
            SiLU(),
            nn.Dense(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ).to_float(self.dtype),
        )
        self.out_layers = nn.SequentialCell(
            GroupNorm(32, self.out_channels),
            SiLU(),
            nn.Dropout(1 - dropout) if is_old_ms_version() else nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(self.out_channels, self.out_channels, 3, pad_mode="pad", padding=1, has_bias=True)
            ).to_float(self.dtype),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, pad_mode="pad", padding=1, has_bias=True
            ).to_float(self.dtype)
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1, has_bias=True).to_float(self.dtype)

        if self.use_temporal_conv:
            self.temporal_conv = TemporalConvBlock_v2(
                self.out_channels, self.out_channels, dropout=0.1, use_image_dataset=use_image_dataset, dtype=self.dtype
            )
            # self.temporal_conv_2 = TemporalConvBlock(self.out_channels, self.out_channels, dropout=0.1, use_image_dataset=use_image_dataset)

    def construct(self, x, emb, batch_size):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).astype(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = ops.chunk(emb_out, 2, axis=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        h = self.skip_connection(x) + h

        if self.use_temporal_conv:
            # (b f) c h w -> b f c h w -> b c f h w
            h = ops.reshape(h, (batch_size, h.shape[0] // batch_size, h.shape[1], h.shape[2], h.shape[3]))
            h = ops.transpose(h, (0, 2, 1, 3, 4))
            h = self.temporal_conv(h)
            # h = self.temporal_conv_2(h)
            # 'b c f h w -> b f c h w -> (b f) c h w
            h = ops.transpose(h, (0, 2, 1, 3, 4))
            h = ops.reshape(h, (-1, h.shape[2], h.shape[3], h.shape[4]))
        return h


class Downsample(nn.Cell):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, padding=1, dtype=ms.float32):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.dtype = dtype
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = nn.Conv2d(
                self.channels, self.out_channels, 3, stride=stride, pad_mode="pad", padding=padding, has_bias=True
            ).to_float(self.dtype)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def construct(self, x):
        # assert x.shape[1] == self.channels
        return self.op(x)


def get_kernel_size_and_stride(input_size, output_size):
    stride = math.floor(input_size / (output_size - 1))
    kernel_size = input_size - (output_size - 1) * stride
    return kernel_size, stride


class UNetSD_temporal(nn.Cell):
    def __init__(
        self,
        cfg,
        in_dim=7,
        dim=512,
        context_dim=512,
        hist_dim=156,
        concat_dim=8,
        out_dim=6,
        dim_mult=[1, 2, 3, 4],
        num_heads=None,
        head_dim=64,
        num_res_blocks=3,
        attn_scales=[1 / 2, 1 / 4, 1 / 8],
        use_scale_shift_norm=True,
        dropout=0.1,
        temporal_attn_times=1,
        temporal_attention=True,
        use_checkpoint=False,
        use_image_dataset=False,
        use_fps_condition=False,
        use_sim_mask=False,
        misc_dropout=0.5,
        inpainting=True,
        video_compositions=["text", "mask"],
        p_all_zero=0.1,
        p_all_keep=0.1,
        use_fp16=False,
        use_adaptive_pool=True,
        use_droppath_masking=True,
        use_recompute=False,
    ):
        self.use_adaptive_pool = use_adaptive_pool
        self.use_droppath_masking = use_droppath_masking

        embed_dim = dim * 4
        num_heads = num_heads if num_heads else dim // 32
        super(UNetSD_temporal, self).__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.cfg = cfg
        self.in_dim = in_dim
        self.dim = dim
        self.context_dim = context_dim
        self.hist_dim = hist_dim
        self.concat_dim = concat_dim
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.dim_mult = dim_mult
        # for temporal attention
        self.num_heads = num_heads
        # for spatial attention
        self.head_dim = head_dim
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.use_scale_shift_norm = use_scale_shift_norm
        self.temporal_attn_times = temporal_attn_times
        self.temporal_attention = temporal_attention
        self.use_checkpoint = use_checkpoint
        self.use_image_dataset = use_image_dataset
        self.use_fps_condition = use_fps_condition
        self.use_sim_mask = use_sim_mask
        self.inpainting = inpainting
        self.video_compositions = video_compositions
        self.p_all_zero = p_all_zero
        self.p_all_keep = p_all_keep
        # self.bernoulli0 = ops.Dropout(keep_prob=p_all_zero) # used to generate zero_mask for droppath on conditions
        # self.bernoulli1= ops.Dropout(keep_prob=p_all_keep)

        use_linear_in_temporal = False
        transformer_depth = 1
        disabled_sa = False
        # params
        enc_dims = [dim * u for u in [1] + dim_mult]
        dec_dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        shortcut_dims = []
        scale = 1.0


        self.encode = nn.SequentialCell(
            nn.Conv2d(4, 128, 3, has_bias=True).to_float(self.dtype),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, has_bias=True).to_float(self.dtype),
            nn.ReLU(),
            nn.Conv2d(128, 4, 3, has_bias=True).to_float(self.dtype),
        )


    def load_state_dict(self, ckpt_path, prefix_to_remove="unet."):
        pass

    def construct(
        self,
        x,
        t,
        y=None,
        depth=None,
        image=None,  # Style Image encoded by clip-vit, shape: [bs, 1, 1024]
        motion=None,
        local_image=None,  # Single Image, i.e. start driving image, shape:
        single_sketch=None,
        masked=None,
        canny=None,
        sketch=None,
        fps=None,
        video_mask=None,
        focus_present_mask=None,
        prob_focus_present=0.0,  # probability at which a given batch sample will focus on the present (0. is all off,
        # 1. is completely arrested attention across time)
        mask_last_frame_num=0,  # mask last frame num
    ):
        assert self.inpainting or masked is None, "inpainting is not supported"
        # start_time = time.time()
        batch, c, f, h, w = x.shape


        # b c f h w -> b f c h w -> (b f) c h w
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        x = ops.reshape(x, (-1, x.shape[2], x.shape[3], x.shape[4]))

        x = self.encode(x)

        # (b f) c h w -> b f c h w -> b c f h w
        x = ops.reshape(x, (batch, x.shape[0] // batch, x.shape[1], x.shape[2], x.shape[3]))
        x = ops.transpose(x, (0, 2, 1, 3, 4))

        return x


if __name__ == "__main__":
    from vc.config.base import cfg

    # [model] unet
    model = UNetSD_temporal(
        cfg,
        in_dim=cfg.unet_in_dim,
        dim=cfg.unet_dim,
        context_dim=cfg.unet_context_dim,
        out_dim=cfg.unet_out_dim,
        dim_mult=cfg.unet_dim_mult,
        num_heads=cfg.unet_num_heads,
        head_dim=cfg.unet_head_dim,
        num_res_blocks=cfg.unet_res_blocks,
        attn_scales=cfg.unet_attn_scales,
        dropout=cfg.unet_dropout,
        temporal_attn_times=0,
        use_checkpoint=cfg.use_checkpoint,
        use_image_dataset=True,
        use_fps_condition=cfg.use_fps_condition,
        video_compositions=cfg.video_compositions,
        use_fp16=True,
    )  # .to(gpu)

    print(int(sum(p.numel() for k, p in model.named_parameters()) / (1024**2)), "M parameters")
