# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import logging

from ..attention import SpatialTransformer
from .util import (
    Identity,
    avg_pool_nd,
    conv_nd,
    linear,
    normalization,
    timestep_embedding,
    zero_module,
)
from .util import is_old_ms_version

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from .motion_module import get_motion_module, VanillaTemporalModule

_logger = logging.getLogger(__name__)


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
        if use_conv:
            self.conv = conv_nd(
                dims, self.channels, self.out_channels, 3, padding=1, has_bias=True, pad_mode="pad"
            ).to_float(dtype)

    def construct(self, x, emb=None, context=None):
        if self.dims == 3:
            x = ops.ResizeNearestNeighbor((x.shape[2] * 2, x.shape[3] * 2, x.shape[4] * 2))(x)
        else:
            x = ops.ResizeNearestNeighbor((x.shape[2] * 2, x.shape[3] * 2))(x)
        if self.use_conv:
            x = self.conv(x)
        return x


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
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=padding, has_bias=True, pad_mode="pad"
            ).to_float(dtype)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def construct(self, x, emb=None, context=None):
        return self.op(x)


class ResBlock(nn.Cell):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout=1.0,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        dtype=ms.float32,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.ori_channels = channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.updown = up or down
        self.dtype = dtype
        self.identity = Identity()
        self.split = ops.Split(1, 2)

        self.in_layers_norm = normalization(channels)
        self.in_layers_silu = nn.SiLU().to_float(self.dtype)
        self.in_layers_conv = conv_nd(
            dims, channels, self.out_channels, 3, padding=1, has_bias=True, pad_mode="pad"
        ).to_float(self.dtype)

        if up:
            self.h_upd = Upsample(channels, False, dims, dtype=self.dtype)
            self.x_upd = Upsample(channels, False, dims, dtype=self.dtype)
        elif down:
            self.h_upd = Downsample(channels, False, dims, dtype=self.dtype)
            self.x_upd = Downsample(channels, False, dims, dtype=self.dtype)
        else:
            self.h_upd = self.x_upd = self.identity

        self.emb_layers = nn.SequentialCell(
            nn.SiLU().to_float(self.dtype),
            linear(
                emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels, dtype=self.dtype
            ),
        )

        self.out_layers_norm = normalization(self.out_channels)
        self.out_layers_silu = nn.SiLU().to_float(self.dtype)

        if is_old_ms_version():
            self.out_layers_drop = nn.Dropout(keep_prob=self.dropout)
        else:
            self.out_layers_drop = nn.Dropout(p=1.0 - self.dropout)

        self.out_layers_conv = zero_module(
            conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1, has_bias=True, pad_mode="pad").to_float(
                self.dtype
            )
        )

        if self.out_channels == channels:
            self.skip_connection = self.identity
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1, has_bias=True, pad_mode="pad"
            ).to_float(self.dtype)
        else:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 1, has_bias=True, pad_mode="pad"
            ).to_float(self.dtype)

    def construct(self, x, emb, context=None):
        if self.updown:
            h = self.in_layers_norm(x)
            h = self.in_layers_silu(h)
            h = self.h_upd(h, emb, context)
            x = self.x_upd(x, emb, context)
            h = self.in_layers_conv(h, emb, context)
        else:
            h = self.in_layers_norm(x)
            h = self.in_layers_silu(h)
            h = self.in_layers_conv(h, emb, context)

        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = ops.expand_dims(emb_out, -1)

        if self.use_scale_shift_norm:
            scale, shift = self.split(emb_out)
            h = self.out_layers_norm(h) * (1 + scale) + shift
            h = self.out_layers_silu(h)
            h = self.out_layers_drop(h)
            h = self.out_layers_conv(h, emb, context)

        else:
            h = h + emb_out
            h = self.out_layers_norm(h)
            h = self.out_layers_silu(h)
            h = self.out_layers_drop(h)
            h = self.out_layers_conv(h, emb, context)

        return self.skip_connection(x) + h


class QKVAttention(nn.Cell):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads


class QKVAttentionLegacy(nn.Cell):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads


class AttentionBlock(nn.Cell):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()


class Timestep(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def construct(self, t):
        return timestep_embedding(t, self.dim)

def rearrange_in(x):
    '''
    reshape x from (b c f h w) -> (b*f c h w)
    temporal 5d to spatial 4d
    '''
    # b c f h w -> b f c h w -> (b f) c h w
    x = ops.transpose(x, (0, 2, 1, 3, 4))
    x = ops.reshape(x, (-1, x.shape[2], x.shape[3], x.shape[4]))
    return x

def rearrange_out(x, f):
    '''
    reshape x from (b*f c h w) -> (b c f h w)
    spatial 4d to temporal 5d
    f: num frames
    '''
    # (b f) c h w -> b f c h w -> b c f h w
    x = ops.reshape(x, (x.shape[0]//f, f, x.shape[1], x.shape[2], x.shape[3]))
    x = ops.transpose(x, (0, 2, 1, 3, 4))
    return x


class UNet3DModel(nn.Cell):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor. channels of image feature encoded by vae, i.e. 4 for SD1.5.
    :param model_channels: base channel count for the model. output channels for conv_in, i.e. 320 for SD1.5
    :param out_channels: channels in the output Tensor. i.e., 4 for SD1.5, same as input.
    :param num_res_blocks: number of residual blocks per downsample. 2 for SD1.5
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used. [ 4, 2, 1 ] for SD1.5
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.  [ 1, 2, 4, 4 ] for SD1.5
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    Additional:

    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        use_linear_in_transformer=False,
        enable_flash_attention=False,
        cross_frame_attention=False,
        unet_chunk_size=2,
        adm_in_channels=None,
        # Additional
        use_inflated_groupnorm=True, # diff, default is to use inference_v2.yaml, more reasonal.
        use_motion_module              = False,
        motion_module_resolutions      = ( 1,2,4,8 ), # used to identify which level to be injected with Motion Module
        motion_module_mid_block        = False,
        motion_module_decoder_only     = False,
        motion_module_type             = None, # default:
        motion_module_kwargs           = {}, #
        unet_use_cross_frame_attention = None,
        unet_use_temporal_attention    = None,
    ):
        super().__init__()

        assert use_inflated_groupnorm==True, "Only support use_inflated_groupnorm=True currently, please use configs/inference/inference_v2.yaml for --inference_config"
        if use_motion_module:
            assert unet_use_cross_frame_attention==False, "not support"
            assert unet_use_temporal_attention==False, "not support"
            assert motion_module_type=='Vanilla', 'not support'
        else:
            print("D---: WARNING: not using motion module")

        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if isinstance(context_dim, ListConfig):
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = 1.0 - dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.SequentialCell(
            linear(model_channels, time_embed_dim, dtype=self.dtype),
            nn.SiLU().to_float(self.dtype),
            linear(time_embed_dim, time_embed_dim, dtype=self.dtype),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim).to_float(self.dtype)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.SequentialCell(
                    nn.SequentialCell(
                        linear(adm_in_channels, time_embed_dim, dtype=self.dtype),
                        nn.SiLU().to_float(self.dtype),
                        linear(time_embed_dim, time_embed_dim, dtype=self.dtype),
                    )
                )
            else:
                raise ValueError("`num_classes` must be an integer or string of 'continuous' or `sequential`")

        self.input_blocks = nn.CellList(
            [
                nn.CellList(
                    [
                        conv_nd(
                            dims, in_channels, model_channels, 3, padding=1, has_bias=True, pad_mode="pad"
                        ).to_float(self.dtype)
                    ]
                )
            ]
        )
        self._feature_size = model_channels #
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        # down blocks, add 2*3+2=8 MotionModules
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                # layers: consist of ResBlock-SptailTrans-MM or ResBlock-MM
                layers = nn.CellList(
                    [
                        ResBlock(
                            ch,
                            time_embed_dim,
                            self.dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            dtype=self.dtype,
                        )
                    ]
                )
                ch = mult * model_channels  # feature channels are doubled in each CrossAttnDownBlock
                # For the first 3 levels, ds=1, 2, 4, create SpatialTransformer. For 4-th level, skip.
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            dtype=self.dtype,
                            dropout=self.dropout,
                            use_linear=use_linear_in_transformer,
                            enable_flash_attention=enable_flash_attention,
                            cross_frame_attention=cross_frame_attention,
                            unet_chunk_size=unet_chunk_size,
                        )
                    )

                # add MotionModule 1) after SpatialTransformer in DownBlockWithAttn, 3*2 times, or 2) after ResBlock in DownBlockWithoutAttn, 1*2 time.
                if use_motion_module:
                    layers.append(
                        get_motion_module( # return VanillaTemporalModule
                            in_channels=ch,
                            motion_module_type=motion_module_type,
                            motion_module_kwargs=motion_module_kwargs,
                        )
                    )

                self.input_blocks.append(layers)
                self._feature_size += ch
                input_block_chans.append(ch)

            # for the first 3 levels, add a DownSampler at each level
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.CellList(
                        [
                            ResBlock(
                                ch,
                                time_embed_dim,
                                self.dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,
                                dtype=self.dtype,
                            )
                        ]
                    )
                    if resblock_updown
                    else nn.CellList([Downsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype)])
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        _logger.debug(
            "Attention: output_channels={}, num_heads={}, num_head_channels:{}, dim_head={}".format(
                ch, num_heads, num_head_channels, dim_head
            )
        )

        # middle block, add 1 MM
        self.middle_block = nn.CellList(
            [
                ResBlock(
                    ch,
                    time_embed_dim,
                    self.dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dtype=self.dtype,
                ),
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                ) if not use_spatial_transformer
                else SpatialTransformer(
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    use_checkpoint=use_checkpoint,
                    dtype=self.dtype,
                    dropout=self.dropout,
                    use_linear=use_linear_in_transformer,
                    enable_flash_attention=enable_flash_attention,
                    cross_frame_attention=cross_frame_attention,
                    unet_chunk_size=unet_chunk_size,
                ),
            ]
        )
        # Add MM after SpatialTrans in MiddleBlock, 1
        if use_motion_module and motion_module_mid_block:
            self.middle_block.append(
                get_motion_module(
                            in_channels=ch,
                            motion_module_type=motion_module_type,
                            motion_module_kwargs=motion_module_kwargs,
                 )
            )
        self.middle_block.append(
                ResBlock(
                    ch,
                    time_embed_dim,
                    self.dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dtype=self.dtype,
                )
        )

        self._feature_size += ch

        # up blocks
        self.output_blocks = nn.CellList([])
        for level, mult in list(enumerate(channel_mult))[::-1]: # run 4 times
            for i in range(num_res_blocks + 1):  # run 3 times
                ich = input_block_chans.pop()
                layers = nn.CellList(
                    [
                        ResBlock(
                            ch + ich,
                            time_embed_dim,
                            self.dropout,
                            out_channels=model_channels * mult,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            dtype=self.dtype,
                        )
                    ]
                )
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            dtype=self.dtype,
                            dropout=self.dropout,
                            use_linear=use_linear_in_transformer,
                            enable_flash_attention=enable_flash_attention,
                            cross_frame_attention=cross_frame_attention,
                            unet_chunk_size=unet_chunk_size,
                        )
                    )

                # Add MM after ResBlock in UpBlockWithoutAttn (1*3), or after SpatialTransformer in UpBlockWithAttn (3*3)
                if use_motion_module:
                    layers.append(
                        get_motion_module(
                            in_channels=ch,
                            motion_module_type=motion_module_type,
                            motion_module_kwargs=motion_module_kwargs,
                        )
                    )

                # Upsample except for the last level
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            dtype=self.dtype,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype)
                    )
                    ds //= 2
                self.output_blocks.append(layers)
                self._feature_size += ch

        self.out = nn.SequentialCell(
            normalization(ch),
            nn.SiLU().to_float(self.dtype),
            zero_module(
                conv_nd(dims, model_channels, out_channels, 3, padding=1, has_bias=True, pad_mode="pad").to_float(
                    self.dtype
                )
            ),
        )

        if self.predict_codebook_ids:
            self.id_predictor = nn.SequentialCell(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1, has_bias=True, pad_mode="pad").to_float(self.dtype),
            )
        self.cat = ops.Concat(axis=1)

    def construct(
        self, x, timesteps=None, context=None, y=None, features_adapter: list = None, append_to_context=None, **kwargs
    ):
        """
        Apply the model to an input batch.
        :param x: (b c f h w), an [N x C x ...] Tensor of inputs.
        :param timesteps: (b,), a 1-D batch of timesteps.
        :param context: (b max_len_tokens dim_ftr), conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: (b c f h w), an [N x C x ...] Tensor of outputs.
        """
        assert len(x.shape)==5, f"UNet3D expect x in shape (b c f h w). but got {x.shape}"
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # time embedding
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        if append_to_context is not None:
            context = ops.cat([context, append_to_context], axis=1)

        # 0. rearrange inputs to (b*f, ...) for pseudo 3d until we meet temporal transformer (i.e. motion module)
        B, C, F, H, W = x.shape
        # x: (b c f h w) -> (b*f c h w)
        x = rearrange_in(x)
        # time mixed with other embedding: (b dim_emb) -> (b*f dim_emb)
        emb = emb.repeat_interleave(repeats=F, dim=0)
        # context: (b max_length dim_clip) -> (b*f dim_emb)
        context = context.repeat_interleave(repeats=F, dim=0)

        h = x

        # 1. conv_in and downblocks
        adapter_idx = 0
        for i, celllist in enumerate(self.input_blocks, 1):
            for cell in celllist:
                if isinstance(cell, VanillaTemporalModule):
                    h = cell(h, temb=emb, encoder_hidden_states=context, video_length=F)
                else:
                    h = cell(h, emb, context)

            if features_adapter and i % 3 == 0:
                h = h + features_adapter[adapter_idx]
                adapter_idx += 1

            hs.append(h)

        if features_adapter:
            assert len(features_adapter) == adapter_idx, "Wrong features_adapter"
        
        # 2. middle block
        for module in self.middle_block:
            # h = module(h, emb, context)
            if isinstance(module, VanillaTemporalModule):
                h = module(h, temb=emb, encoder_hidden_states=context, video_length=F)
            else:
                h = module(h, emb, context)

        # 3. up blocks
        hs_index = -1
        for celllist in self.output_blocks:
            h = self.cat((h, hs[hs_index]))
            for cell in celllist:
                # h = cell(h, emb, context)
                if isinstance(cell, VanillaTemporalModule):
                    h = cell(h, temb=emb, encoder_hidden_states=context, video_length=F)
                else:
                    h = cell(h, emb, context)
            hs_index -= 1

        if self.predict_codebook_ids:
            assert self.predict_codebook_ids, "Not impl yet"
            return self.id_predictor(h)
        else:
            h = self.out(h)

            # rearrange back: (b*f c h w) -> (b c f h w)
            h = rearrange_out(h, f=F)

            return h