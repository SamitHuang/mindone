# diffusers/models/autoencoders/autoencoder_kl_cogvideox.py -- v0.31.0

import logging
import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
from safetensors import safe_open

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.nn.functional as F
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor

from .layer import GroupNorm

__all__ = ["AutoencoderKLCogVideoX", "CogVideoX_VAE"]

logger = logging.getLogger(__name__)


ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "mish": nn.Mish,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


def get_activation(act_fn: str) -> nn.Cell:
    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]()
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


class FP32GroupNorm(GroupNorm):
    def construct(self, input: Tensor) -> Tensor:
        dtype = input.dtype
        y = ops.group_norm(
            input.to(ms.float32), self.num_groups, self.weight.to(ms.float32), self.bias.to(ms.float32), self.eps
        ).to(dtype)
        return y


class CogVideoXSafeConv3d(nn.Conv3d):
    def construct(self, input: Tensor) -> Tensor:
        memory_count = (
            (input.shape[0] * input.shape[1] * input.shape[2] * input.shape[3] * input.shape[4]) * 2 / 1024**3
        )

        # Set to 2GB
        if memory_count > 2:
            kernel_size = self.kernel_size[0]
            part_num = int(memory_count / 2) + 1
            input_chunks = mint.chunk(input, part_num, dim=2)

            if kernel_size > 1:
                input_chunks = [input_chunks[0]] + [
                    mint.cat((input_chunks[i - 1][:, :, -kernel_size + 1 :], input_chunks[i]), dim=2)
                    for i in range(1, len(input_chunks))
                ]

            output_chunks = []
            for input_chunk in input_chunks:
                output_chunks.append(self._construct(input_chunk))
            output = mint.cat(output_chunks, dim=2)
            return output
        else:
            return self._construct(input)

    def _construct(self, x):
        # TODO: drop cast when it supports fp32
        dtype = x.dtype
        assert self.group == 1
        out = self.conv3d(x.to(ms.bfloat16), self.weight.to(ms.bfloat16))
        if self.has_bias:
            out = self.bias_add(out, self.bias.to(ms.bfloat16))
        return out.to(dtype)


class CogVideoXCausalConv3d(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: int = 1,
        dilation: int = 1,
        pad_mode: str = "constant",
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 3

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.height_pad = height_pad
        self.width_pad = width_pad
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad, height_pad, time_pad, 0)

        self.temporal_dim = 2
        self.time_kernel_size = time_kernel_size

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = CogVideoXSafeConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode="pad",
            dilation=dilation,
            has_bias=True,
            dtype=dtype,
        )

    def fake_context_parallel_forward(self, inputs: Tensor, conv_cache: Optional[Tensor] = None) -> Tensor:
        kernel_size = self.time_kernel_size
        if kernel_size > 1:
            cached_inputs = [conv_cache] if conv_cache is not None else [inputs[:, :, :1]] * (kernel_size - 1)
            inputs = mint.cat(cached_inputs + [inputs], dim=2)
        return inputs

    def construct(self, inputs: Tensor, conv_cache: Optional[Tensor] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        inputs = self.fake_context_parallel_forward(inputs, conv_cache)
        conv_cache = inputs[:, :, -self.time_kernel_size + 1 :].copy()

        padding_2d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad)
        inputs = F.pad(inputs, padding_2d, mode="constant", value=0)

        output = self.conv(inputs)
        return output, conv_cache


class CogVideoXSpatialNorm3D(nn.Cell):
    def __init__(
        self,
        f_channels: int,
        zq_channels: int,
        groups: int = 32,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.norm_layer = FP32GroupNorm(num_channels=f_channels, num_groups=groups, eps=1e-6, affine=True, dtype=dtype)
        self.conv_y = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1, dtype=dtype)
        self.conv_b = CogVideoXCausalConv3d(zq_channels, f_channels, kernel_size=1, stride=1, dtype=dtype)

    def construct(
        self, f: Tensor, zq: Tensor, conv_cache: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        if f.shape[2] > 1 and f.shape[2] % 2 == 1:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            z_first, z_rest = zq[:, :, :1], zq[:, :, 1:]
            z_first = F.interpolate(z_first, size=f_first_size)
            z_rest = F.interpolate(z_rest, size=f_rest_size)
            zq = mint.cat([z_first, z_rest], dim=2)
        else:
            zq = F.interpolate(zq, size=f.shape[-3:])

        conv_y, new_conv_cache["conv_y"] = self.conv_y(zq, conv_cache=conv_cache.get("conv_y"))
        conv_b, new_conv_cache["conv_b"] = self.conv_b(zq, conv_cache=conv_cache.get("conv_b"))

        norm_f = self.norm_layer(f)
        new_f = norm_f * conv_y + conv_b
        return new_f, new_conv_cache


class CogVideoXResnetBlock3D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        conv_shortcut: bool = False,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nonlinearity = get_activation(non_linearity)
        self.use_conv_shortcut = conv_shortcut
        self.spatial_norm_dim = spatial_norm_dim

        if spatial_norm_dim is None:
            self.norm1 = FP32GroupNorm(num_channels=in_channels, num_groups=groups, eps=eps, dtype=dtype)
            self.norm2 = FP32GroupNorm(num_channels=out_channels, num_groups=groups, eps=eps, dtype=dtype)
        else:
            self.norm1 = CogVideoXSpatialNorm3D(
                f_channels=in_channels, zq_channels=spatial_norm_dim, groups=groups, dtype=dtype
            )
            self.norm2 = CogVideoXSpatialNorm3D(
                f_channels=out_channels, zq_channels=spatial_norm_dim, groups=groups, dtype=dtype
            )

        self.conv1 = CogVideoXCausalConv3d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode, dtype=dtype
        )

        if temb_channels > 0:
            self.temb_proj = mint.nn.Linear(in_features=temb_channels, out_features=out_channels, dtype=dtype)

        self.dropout = mint.nn.Dropout(dropout)
        self.conv2 = CogVideoXCausalConv3d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode, dtype=dtype
        )

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CogVideoXCausalConv3d(
                    in_channels=in_channels, out_channels=out_channels, kernel_size=3, pad_mode=pad_mode, dtype=dtype
                )
            else:
                self.conv_shortcut = CogVideoXSafeConv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    pad_mode="pad",
                    padding=0,
                    has_bias=True,
                    dtype=dtype,
                )

    def construct(
        self,
        inputs: Tensor,
        temb: Optional[Tensor] = None,
        zq: Optional[Tensor] = None,
        conv_cache: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ) -> Tuple[Tensor, Dict[str, Dict[str, Tensor]]]:
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states = inputs

        if zq is not None:
            hidden_states, new_conv_cache["norm1"] = self.norm1(hidden_states, zq, conv_cache=conv_cache.get("norm1"))
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states, new_conv_cache["conv1"] = self.conv1(hidden_states, conv_cache=conv_cache.get("conv1"))

        if temb is not None:
            hidden_states = hidden_states + self.temb_proj(self.nonlinearity(temb))[:, :, None, None, None]

        if zq is not None:
            hidden_states, new_conv_cache["norm2"] = self.norm2(hidden_states, zq, conv_cache=conv_cache.get("norm2"))
        else:
            hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states, new_conv_cache["conv2"] = self.conv2(hidden_states, conv_cache=conv_cache.get("conv2"))

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                inputs, new_conv_cache["conv_shortcut"] = self.conv_shortcut(
                    inputs, conv_cache=conv_cache.get("conv_shortcut")
                )
            else:
                inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states, new_conv_cache


class CogVideoXDownsample3D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 0,
        compress_time: bool = False,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode="pad",
            padding=padding,
            has_bias=True,
            dtype=dtype,
        )
        self.compress_time = compress_time

    def construct(self, x: Tensor) -> Tensor:
        if self.compress_time:
            batch_size, channels, frames, height, width = x.shape

            # (batch_size, channels, frames, height, width) -> (batch_size, height, width, channels, frames) -> (batch_size * height * width, channels, frames)
            x = x.permute(0, 3, 4, 1, 2).reshape(batch_size * height * width, channels, frames)

            if x.shape[-1] % 2 == 1:
                x_first, x_rest = x[..., 0], x[..., 1:]
                if x_rest.shape[-1] > 0:
                    # (batch_size * height * width, channels, frames - 1) -> (batch_size * height * width, channels, (frames - 1) // 2)
                    # TODO: change to avg_pool1d once it is supported
                    # x_rest = F.avg_pool1d(x_rest, kernel_size=2, stride=2)
                    x_rest = F.avg_pool2d(x_rest.unsqueeze(-1), kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)

                x = mint.cat([x_first[..., None], x_rest], dim=-1)
                # (batch_size * height * width, channels, (frames // 2) + 1) -> (batch_size, height, width, channels, (frames // 2) + 1)
                # -> (batch_size, channels, (frames // 2) + 1, height, width)
                x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)
            else:
                # (batch_size * height * width, channels, frames) -> (batch_size * height * width, channels, frames // 2)
                # TODO: change to avg_pool1d once it is supported
                # x = F.avg_pool1d(x, kernel_size=2, stride=2)
                x = F.avg_pool2d(x.unsqueeze(-1), kernel_size=(2, 1), stride=(2, 1)).squeeze(-1)
                # (batch_size * height * width, channels, frames // 2) -> (batch_size, height, width, channels, frames // 2)
                # -> (batch_size, channels, frames // 2, height, width)
                x = x.reshape(batch_size, height, width, channels, x.shape[-1]).permute(0, 3, 4, 1, 2)

        # Pad the tensor
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        batch_size, channels, frames, height, width = x.shape
        # (batch_size, channels, frames, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size * frames, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        x = self.conv(x)
        # (batch_size * frames, channels, height, width) -> (batch_size, frames, channels, height, width) -> (batch_size, channels, frames, height, width)
        x = x.reshape(batch_size, frames, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        return x


class CogVideoXDownBlock3D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_padding: int = 0,
        compress_time: bool = False,
        pad_mode: str = "first",
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    pad_mode=pad_mode,
                    dtype=dtype,
                )
            )

        self.resnets = nn.CellList(resnets)
        self.downsamplers = None

        if add_downsample:
            self.downsamplers = nn.CellList(
                [
                    CogVideoXDownsample3D(
                        out_channels, out_channels, padding=downsample_padding, compress_time=compress_time, dtype=dtype
                    )
                ]
            )

    def construct(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None,
        zq: Optional[Tensor] = None,
        conv_cache: Optional[Dict[str, Dict[str, Dict[str, Tensor]]]] = None,
    ) -> Tuple[Tensor, Dict[str, Dict[str, Dict[str, Tensor]]]]:
        r"""Forward method of the `CogVideoXDownBlock3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"

            hidden_states, new_conv_cache[conv_cache_key] = resnet(
                hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
            )

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states, new_conv_cache


class CogVideoXMidBlock3D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatial_norm_dim: Optional[int] = None,
        pad_mode: str = "first",
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    spatial_norm_dim=spatial_norm_dim,
                    non_linearity=resnet_act_fn,
                    pad_mode=pad_mode,
                    dtype=dtype,
                )
            )
        self.resnets = nn.CellList(resnets)

    def construct(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None,
        zq: Optional[Tensor] = None,
        conv_cache: Optional[Dict[str, Dict[str, Dict[str, Tensor]]]] = None,
    ) -> Tuple[Tensor, Dict[str, Dict[str, Dict[str, Tensor]]]]:
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"

            hidden_states, new_conv_cache[conv_cache_key] = resnet(
                hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
            )

        return hidden_states, new_conv_cache


class CogVideoXUpsample3D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        compress_time: bool = False,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            pad_mode="pad",
            padding=padding,
            has_bias=True,
            dtype=dtype,
        )
        self.compress_time = compress_time

    def construct(self, inputs: Tensor) -> Tensor:
        if self.compress_time:
            if inputs.shape[2] > 1 and inputs.shape[2] % 2 == 1:
                # split first frame
                x_first, x_rest = inputs[:, :, 0], inputs[:, :, 1:]

                x_first = F.interpolate(x_first, scale_factor=2.0)
                x_rest = F.interpolate(x_rest, scale_factor=2.0)
                x_first = x_first[:, :, None, :, :]
                inputs = mint.cat([x_first, x_rest], dim=2)
            elif inputs.shape[2] > 1:
                inputs = F.interpolate(inputs, scale_factor=2.0)
            else:
                inputs = inputs.squeeze(2)
                inputs = F.interpolate(inputs, scale_factor=2.0)
                inputs = inputs[:, :, None, :, :]
        else:
            # only interpolate 2D
            b, c, t, h, w = inputs.shape
            inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
            inputs = F.interpolate(inputs, scale_factor=2.0)
            inputs = inputs.reshape(b, t, c, *inputs.shape[2:]).permute(0, 2, 1, 3, 4)

        b, c, t, h, w = inputs.shape
        inputs = inputs.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        inputs = self.conv(inputs)
        inputs = inputs.reshape(b, t, *inputs.shape[1:]).permute(0, 2, 1, 3, 4)

        return inputs


class CogVideoXUpBlock3D(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        spatial_norm_dim: int = 16,
        add_upsample: bool = True,
        upsample_padding: int = 1,
        compress_time: bool = False,
        pad_mode: str = "first",
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        resnets = []
        for i in range(num_layers):
            in_channel = in_channels if i == 0 else out_channels
            resnets.append(
                CogVideoXResnetBlock3D(
                    in_channels=in_channel,
                    out_channels=out_channels,
                    dropout=dropout,
                    temb_channels=temb_channels,
                    groups=resnet_groups,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    spatial_norm_dim=spatial_norm_dim,
                    pad_mode=pad_mode,
                    dtype=dtype,
                )
            )

        self.resnets = nn.CellList(resnets)
        self.upsamplers = None

        if add_upsample:
            self.upsamplers = nn.CellList(
                [
                    CogVideoXUpsample3D(
                        out_channels, out_channels, padding=upsample_padding, compress_time=compress_time, dtype=dtype
                    )
                ]
            )

    def construct(
        self,
        hidden_states: Tensor,
        temb: Optional[Tensor] = None,
        zq: Optional[Tensor] = None,
        conv_cache: Optional[Dict[str, Dict[str, Dict[str, Tensor]]]] = None,
    ) -> Tuple[Tensor, Dict[str, Dict[str, Dict[str, Tensor]]]]:
        r"""Forward method of the `CogVideoXUpBlock3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        for i, resnet in enumerate(self.resnets):
            conv_cache_key = f"resnet_{i}"

            hidden_states, new_conv_cache[conv_cache_key] = resnet(
                hidden_states, temb, zq, conv_cache=conv_cache.get(conv_cache_key)
            )

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states, new_conv_cache


class CogVideoXEncoder3D(nn.Cell):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        down_block_types: Tuple[str, ...] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "first",
        temporal_compression_ratio: float = 4,
        dtype: ms.Type = ms.float32,
    ):
        super().__init__()

        # log2 of temporal_compress_times
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        self.conv_in = CogVideoXCausalConv3d(
            in_channels, block_out_channels[0], kernel_size=3, pad_mode=pad_mode, dtype=dtype
        )
        self.down_blocks = nn.CellList([])

        # down blocks
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            if down_block_type == "CogVideoXDownBlock3D":
                down_block = CogVideoXDownBlock3D(
                    in_channels=input_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    dropout=dropout,
                    num_layers=layers_per_block,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    add_downsample=not is_final_block,
                    compress_time=compress_time,
                    dtype=dtype,
                )
            else:
                raise ValueError("Invalid `down_block_type` encountered. Must be `CogVideoXDownBlock3D`")

            self.down_blocks.append(down_block)

        # mid block
        self.mid_block = CogVideoXMidBlock3D(
            in_channels=block_out_channels[-1],
            temb_channels=0,
            dropout=dropout,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            pad_mode=pad_mode,
            dtype=dtype,
        )

        self.norm_out = FP32GroupNorm(norm_num_groups, block_out_channels[-1], eps=1e-6, dtype=dtype)
        self.conv_act = nn.SiLU()
        self.conv_out = CogVideoXCausalConv3d(
            block_out_channels[-1], 2 * out_channels, kernel_size=3, pad_mode=pad_mode, dtype=dtype
        )

    def construct(
        self,
        sample: Tensor,
        temb: Optional[Tensor] = None,
        conv_cache: Optional[Dict[str, Dict[str, Union[Tensor, Dict[str, Dict[str, Tensor]]]]]] = None,
    ) -> Tuple[Tensor, Dict[str, Dict[str, Union[Tensor, Dict[str, Dict[str, Tensor]]]]]]:
        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states, new_conv_cache["conv_in"] = self.conv_in(sample, conv_cache=conv_cache.get("conv_in"))

        # 1. Down
        for i, down_block in enumerate(self.down_blocks):
            conv_cache_key = f"down_block_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = down_block(
                hidden_states, temb, None, conv_cache=conv_cache.get(conv_cache_key)
            )

        # 2. Mid
        hidden_states, new_conv_cache["mid_block"] = self.mid_block(
            hidden_states, temb, None, conv_cache=conv_cache.get("mid_block")
        )

        # 3. Post-process
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)

        hidden_states, new_conv_cache["conv_out"] = self.conv_out(hidden_states, conv_cache=conv_cache.get("conv_out"))

        return hidden_states, new_conv_cache


class CogVideoXDecoder3D(nn.Cell):
    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 256, 512),
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        dropout: float = 0.0,
        pad_mode: str = "first",
        temporal_compression_ratio: float = 4,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        reversed_block_out_channels = list(reversed(block_out_channels))

        self.conv_in = CogVideoXCausalConv3d(
            in_channels, reversed_block_out_channels[0], kernel_size=3, pad_mode=pad_mode, dtype=dtype
        )

        # mid block
        self.mid_block = CogVideoXMidBlock3D(
            in_channels=reversed_block_out_channels[0],
            temb_channels=0,
            num_layers=2,
            resnet_eps=norm_eps,
            resnet_act_fn=act_fn,
            resnet_groups=norm_num_groups,
            spatial_norm_dim=in_channels,
            pad_mode=pad_mode,
            dtype=dtype,
        )

        # up blocks
        self.up_blocks = nn.CellList([])

        output_channel = reversed_block_out_channels[0]
        temporal_compress_level = int(np.log2(temporal_compression_ratio))

        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            compress_time = i < temporal_compress_level

            if up_block_type == "CogVideoXUpBlock3D":
                up_block = CogVideoXUpBlock3D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    temb_channels=0,
                    dropout=dropout,
                    num_layers=layers_per_block + 1,
                    resnet_eps=norm_eps,
                    resnet_act_fn=act_fn,
                    resnet_groups=norm_num_groups,
                    spatial_norm_dim=in_channels,
                    add_upsample=not is_final_block,
                    compress_time=compress_time,
                    pad_mode=pad_mode,
                    dtype=dtype,
                )
                prev_output_channel = output_channel
            else:
                raise ValueError("Invalid `up_block_type` encountered. Must be `CogVideoXUpBlock3D`")

            self.up_blocks.append(up_block)

        self.norm_out = CogVideoXSpatialNorm3D(
            reversed_block_out_channels[-1], in_channels, groups=norm_num_groups, dtype=dtype
        )
        self.conv_act = nn.SiLU()
        self.conv_out = CogVideoXCausalConv3d(
            reversed_block_out_channels[-1], out_channels, kernel_size=3, pad_mode=pad_mode, dtype=dtype
        )

    def construct(
        self,
        sample: Tensor,
        temb: Optional[Tensor] = None,
        conv_cache: Optional[Dict[str, Dict[str, Union[Tensor, Dict[str, Dict[str, Tensor]]]]]] = None,
    ) -> Tuple[Tensor, Dict[str, Dict[str, Union[Tensor, Dict[str, Dict[str, Tensor]]]]]]:
        r"""The forward method of the `CogVideoXDecoder3D` class."""

        new_conv_cache = {}
        conv_cache = conv_cache or {}

        hidden_states, new_conv_cache["conv_in"] = self.conv_in(sample, conv_cache=conv_cache.get("conv_in"))

        # 1. Mid
        hidden_states, new_conv_cache["mid_block"] = self.mid_block(
            hidden_states, temb, sample, conv_cache=conv_cache.get("mid_block")
        )

        # 2. Up
        for i, up_block in enumerate(self.up_blocks):
            conv_cache_key = f"up_block_{i}"
            hidden_states, new_conv_cache[conv_cache_key] = up_block(
                hidden_states, temb, sample, conv_cache=conv_cache.get(conv_cache_key)
            )

        # 3. Post-process
        hidden_states, new_conv_cache["norm_out"] = self.norm_out(
            hidden_states, sample, conv_cache=conv_cache.get("norm_out")
        )
        hidden_states = self.conv_act(hidden_states)
        hidden_states, new_conv_cache["conv_out"] = self.conv_out(hidden_states, conv_cache=conv_cache.get("conv_out"))

        return hidden_states, new_conv_cache


class AutoencoderKLCogVideoX(nn.Cell):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = (
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
            "CogVideoXDownBlock3D",
        ),
        up_block_types: Tuple[str] = (
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
            "CogVideoXUpBlock3D",
        ),
        block_out_channels: Tuple[int] = (128, 256, 256, 512),
        latent_channels: int = 16,
        layers_per_block: int = 3,
        act_fn: str = "silu",
        norm_eps: float = 1e-6,
        norm_num_groups: int = 32,
        temporal_compression_ratio: float = 4,
        sample_height: int = 480,
        sample_width: int = 720,
        scaling_factor: float = 1.15258426,
        shift_factor: Optional[float] = None,
        latents_mean: Optional[Tuple[float]] = None,
        latents_std: Optional[Tuple[float]] = None,
        force_upcast: float = True,
        use_quant_conv: bool = False,
        use_post_quant_conv: bool = False,
        dtype: ms.Type = ms.float32,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels
        self.patch_size = (
            int(temporal_compression_ratio),
            int(2 ** (len(block_out_channels) - 1)),
            int(2 ** (len(block_out_channels) - 1)),
        )
        self.scaling_factor = scaling_factor

        self.block_out_channels = block_out_channels

        self.encoder = CogVideoXEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
            dtype=dtype,
        )
        self.decoder = CogVideoXDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_eps=norm_eps,
            norm_num_groups=norm_num_groups,
            temporal_compression_ratio=temporal_compression_ratio,
            dtype=dtype,
        )
        self.quant_conv = (
            CogVideoXSafeConv3d(2 * out_channels, 2 * out_channels, 1, pad_mode="pad", has_bias=True, dtype=dtype)
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            CogVideoXSafeConv3d(out_channels, out_channels, 1, pad_mode="pad", has_bias=True, dtype=dtype)
            if use_post_quant_conv
            else None
        )

        self.use_slicing = False
        self.use_tiling = False

        # Can be increased to decode more latent frames at once, but comes at a reasonable memory cost and it is not
        # recommended because the temporal parts of the VAE, here, are tricky to understand.
        # If you decode X latent frames together, the number of output frames is:
        #     (X + (2 conv cache) + (2 time upscale_1) + (4 time upscale_2) - (2 causal conv downscale)) => X + 6 frames
        #
        # Example with num_latent_frames_batch_size = 2:
        #     - 12 latent frames: (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11) are processed together
        #         => (12 // 2 frame slices) * ((2 num_latent_frames_batch_size)
        #             + (2 conv cache) + (2 time upscale_1) + (4 time upscale_2) - (2 causal conv downscale))
        #         => 6 * 8 = 48 frames
        #     - 13 latent frames: (0, 1, 2) (special case), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12) are processed together
        #         => (1 frame slice) * ((3 num_latent_frames_batch_size) + (2 conv cache) + (2 time upscale_1) + (4 time upscale_2) - (2 causal conv downscale))
        #             +
        #            ((13 - 3) // 2) * ((2 num_latent_frames_batch_size) + (2 conv cache) + (2 time upscale_1) + (4 time upscale_2) - (2 causal conv downscale))
        #         => 1 * 9 + 5 * 8 = 49 frames
        # It has been implemented this way so as to not have "magic values" in the code base that would be hard to explain. Note that
        # setting it to anything other than 2 would give poor results because the VAE hasn't been trained to be adaptive with different
        # number of temporal frames.
        self.num_latent_frames_batch_size = 2
        self.num_sample_frames_batch_size = 8

        # We make the minimum height and width of sample for tiling half that of the generally supported
        self.tile_sample_min_height = sample_height // 2
        self.tile_sample_min_width = sample_width // 2
        self.tile_latent_min_height = int(self.tile_sample_min_height / (2 ** (len(self.block_out_channels) - 1)))
        self.tile_latent_min_width = int(self.tile_sample_min_width / (2 ** (len(self.block_out_channels) - 1)))

        # These are experimental overlap factors that were chosen based on experimentation and seem to work best for
        # 720x480 (WxH) resolution. The above resolution is the strongly recommended generation resolution in CogVideoX
        # and so the tiling implementation has only been tested on those specific resolutions.
        self.tile_overlap_factor_height = 1 / 6
        self.tile_overlap_factor_width = 1 / 5
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    def enable_tiling(
        self,
        tile_sample_min_height: Optional[int] = None,
        tile_sample_min_width: Optional[int] = None,
        tile_overlap_factor_height: Optional[float] = None,
        tile_overlap_factor_width: Optional[float] = None,
    ) -> None:
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_latent_min_height = int(self.tile_sample_min_height / (2 ** (len(self.block_out_channels) - 1)))
        self.tile_latent_min_width = int(self.tile_sample_min_width / (2 ** (len(self.block_out_channels) - 1)))
        self.tile_overlap_factor_height = tile_overlap_factor_height or self.tile_overlap_factor_height
        self.tile_overlap_factor_width = tile_overlap_factor_width or self.tile_overlap_factor_width

    def disable_tiling(self) -> None:
        self.use_tiling = False

    def enable_slicing(self) -> None:
        self.use_slicing = True

    def disable_slicing(self) -> None:
        self.use_slicing = False

    def _encode(self, x: Tensor) -> Tensor:
        batch_size, num_channels, num_frames, height, width = x.shape

        if self.use_tiling and (width > self.tile_sample_min_width or height > self.tile_sample_min_height):
            return self.tiled_encode(x)

        frame_batch_size = self.num_sample_frames_batch_size
        # Note: We expect the number of frames to be either `1` or `frame_batch_size * k` or `frame_batch_size * k + 1` for some k.
        # As the extra single frame is handled inside the loop, it is not required to round up here.
        num_batches = max(num_frames // frame_batch_size, 1)
        conv_cache = None
        enc = []

        for i in range(num_batches):
            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            x_intermediate = x[:, :, start_frame:end_frame]
            x_intermediate, conv_cache = self.encoder(x_intermediate, conv_cache=conv_cache)
            if self.quant_conv is not None:
                x_intermediate = self.quant_conv(x_intermediate)
            enc.append(x_intermediate)

        enc = mint.cat(enc, dim=2)
        return enc

    def encode(self, x: Tensor) -> Tensor:
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = mint.cat(encoded_slices)
        else:
            h = self._encode(x)

        return h

    def encode_with_moments_output(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """For latent caching usage"""
        x = self.encode(x)
        mean, logvar = mint.chunk(x, 2, dim=1)
        logvar = mint.clamp(logvar, -30.0, 20.0)
        std = mint.exp(0.5 * logvar)
        return mean, std

    def _decode(self, z: Tensor) -> Tensor:
        batch_size, num_channels, num_frames, height, width = z.shape

        if self.use_tiling and (width > self.tile_latent_min_width or height > self.tile_latent_min_height):
            return self.tiled_decode(z)

        frame_batch_size = self.num_latent_frames_batch_size
        num_batches = max(num_frames // frame_batch_size, 1)
        conv_cache = None
        dec = []

        for i in range(num_batches):
            remaining_frames = num_frames % frame_batch_size
            start_frame = frame_batch_size * i + (0 if i == 0 else remaining_frames)
            end_frame = frame_batch_size * (i + 1) + remaining_frames
            z_intermediate = z[:, :, start_frame:end_frame]
            if self.post_quant_conv is not None:
                z_intermediate = self.post_quant_conv(z_intermediate)
            z_intermediate, conv_cache = self.decoder(z_intermediate, conv_cache=conv_cache)
            dec.append(z_intermediate)

        dec = mint.cat(dec, dim=2)

        return dec

    def decode(self, z: Tensor) -> Tensor:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice) for z_slice in z.split(1)]
            decoded = mint.cat(decoded_slices)
        else:
            decoded = self._decode(z)

        return decoded

    def blend_v(self, a: Tensor, b: Tensor, blend_extent: int) -> Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (
                y / blend_extent
            )
        return b

    def blend_h(self, a: Tensor, b: Tensor, blend_extent: int) -> Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (
                x / blend_extent
            )
        return b

    def tiled_encode(self, x: Tensor) -> Tensor:
        # For a rough memory estimate, take a look at the `tiled_decode` method.
        batch_size, num_channels, num_frames, height, width = x.shape

        overlap_height = int(self.tile_sample_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_sample_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_latent_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_latent_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_latent_min_height - blend_extent_height
        row_limit_width = self.tile_latent_min_width - blend_extent_width
        frame_batch_size = self.num_sample_frames_batch_size

        # Split x into overlapping tiles and encode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                # Note: We expect the number of frames to be either `1` or `frame_batch_size * k` or `frame_batch_size * k + 1` for some k.
                # As the extra single frame is handled inside the loop, it is not required to round up here.
                num_batches = max(num_frames // frame_batch_size, 1)
                conv_cache = None
                time = []

                for k in range(num_batches):
                    remaining_frames = num_frames % frame_batch_size
                    start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                    end_frame = frame_batch_size * (k + 1) + remaining_frames
                    tile = x[
                        :,
                        :,
                        start_frame:end_frame,
                        i : i + self.tile_sample_min_height,
                        j : j + self.tile_sample_min_width,
                    ]
                    tile, conv_cache = self.encoder(tile, conv_cache=conv_cache)
                    if self.quant_conv is not None:
                        tile = self.quant_conv(tile)
                    time.append(tile)

                row.append(mint.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(mint.cat(result_row, dim=4))

        enc = mint.cat(result_rows, dim=3)
        return enc

    def tiled_decode(self, z: Tensor) -> Tensor:
        # Rough memory assessment:
        #   - In CogVideoX-2B, there are a total of 24 CausalConv3d layers.
        #   - The biggest intermediate dimensions are: [1, 128, 9, 480, 720].
        #   - Assume fp16 (2 bytes per value).
        # Memory required: 1 * 128 * 9 * 480 * 720 * 24 * 2 / 1024**3 = 17.8 GB
        #
        # Memory assessment when using tiling:
        #   - Assume everything as above but now HxW is 240x360 by tiling in half
        # Memory required: 1 * 128 * 9 * 240 * 360 * 24 * 2 / 1024**3 = 4.5 GB

        batch_size, num_channels, num_frames, height, width = z.shape

        overlap_height = int(self.tile_latent_min_height * (1 - self.tile_overlap_factor_height))
        overlap_width = int(self.tile_latent_min_width * (1 - self.tile_overlap_factor_width))
        blend_extent_height = int(self.tile_sample_min_height * self.tile_overlap_factor_height)
        blend_extent_width = int(self.tile_sample_min_width * self.tile_overlap_factor_width)
        row_limit_height = self.tile_sample_min_height - blend_extent_height
        row_limit_width = self.tile_sample_min_width - blend_extent_width
        frame_batch_size = self.num_latent_frames_batch_size

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, height, overlap_height):
            row = []
            for j in range(0, width, overlap_width):
                num_batches = max(num_frames // frame_batch_size, 1)
                conv_cache = None
                time = []

                for k in range(num_batches):
                    remaining_frames = num_frames % frame_batch_size
                    start_frame = frame_batch_size * k + (0 if k == 0 else remaining_frames)
                    end_frame = frame_batch_size * (k + 1) + remaining_frames
                    tile = z[
                        :,
                        :,
                        start_frame:end_frame,
                        i : i + self.tile_latent_min_height,
                        j : j + self.tile_latent_min_width,
                    ]
                    if self.post_quant_conv is not None:
                        tile = self.post_quant_conv(tile)
                    tile, conv_cache = self.decoder(tile, conv_cache=conv_cache)
                    time.append(tile)

                row.append(mint.cat(time, dim=2))
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent_width)
                result_row.append(tile[:, :, :, :row_limit_height, :row_limit_width])
            result_rows.append(mint.cat(result_row, dim=4))

        dec = mint.cat(result_rows, dim=3)

        return dec

    def sample(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        mean, logvar = mint.chunk(x, 2, dim=1)
        logvar = mint.clamp(logvar, -30.0, 20.0)
        std = mint.exp(0.5 * logvar)
        x = mint.normal(mean=mean.to(ms.float32), std=std.to(ms.float16), size=mean.shape).to(dtype)
        return x

    def mode(self, x: Tensor) -> Tensor:
        mean, _ = mint.chunk(x, 2, dim=1)
        return mean

    def construct(
        self,
        sample: Tensor,
        sample_posterior: bool = False,
    ) -> Tensor:
        x = sample
        posterior = self.encode(x)
        if sample_posterior:
            z = self.sample(posterior)
        else:
            z = self.mode(posterior)
        dec = self.decode(z)
        return dec

    def load_from_checkpoint(self, ckpt_path: str) -> None:
        if not os.path.exists(ckpt_path):
            logger.warning(f"{ckpt_path} not found. No checkpoint loaded!!")
        else:
            format = "safetensors" if ckpt_path.endswith(".safetensors") else "ckpt"
            if format == "ckpt":
                parameter_dict = ms.load_checkpoint(ckpt_path)
            else:
                parameter_dict = dict()
                with safe_open(ckpt_path, framework="np") as f:
                    for k in f.keys():
                        parameter_dict[k] = Parameter(Tensor(f.get_tensor(k), dtype=self.dtype))
            param_not_load, ckpt_not_load = ms.load_param_into_net(self, parameter_dict, strict_load=True)
            assert len(param_not_load) == 0 and len(ckpt_not_load) == 0

    def get_latent_size(self, input_size: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
        latent_size = []
        for i in range(3):
            if input_size[i] is None:
                latent_size.append(None)
            else:
                if i == 0:
                    latent_size.append((input_size[i] - 1) // self.patch_size[i] + 1)
                else:
                    latent_size.append(input_size[i] // self.patch_size[i])
        return latent_size


def CogVideoX_VAE(from_pretrained: Optional[str] = None, **kwargs) -> AutoencoderKLCogVideoX:
    model = AutoencoderKLCogVideoX(**kwargs)

    if from_pretrained is not None:
        model.load_from_checkpoint(from_pretrained)
    return model
