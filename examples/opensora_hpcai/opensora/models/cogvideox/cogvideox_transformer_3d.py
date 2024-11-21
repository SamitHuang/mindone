# diffusers/models/transformers/cogvideox_transformer_3d.py -- v0.31.0
import logging
from typing import List, Literal, Optional, Tuple, Union
import math
import numbers

import numpy as np
from opensora.acceleration.communications import AlltoAll, GatherFowardSplitBackward, SplitFowardGatherBackward
from opensora.acceleration.parallel_states import get_sequence_parallel_group
from safetensors import safe_open

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.nn.functional as F
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor

from mindspore.common.initializer import initializer
from mindspore.communication import get_group_size

from mindspore.ops.operations.nn_ops import FlashAttentionScore

__all__ = ["CogVideoXTransformer3DModel", "CogVideoX_2B", "CogVideoX_5B", "CogVideoX_5B_v1_5"]

logger = logging.getLogger(__name__)


def apply_rotary_emb(x: Tensor, freqs_cis: Tensor) -> Tensor:
    cos, sin = freqs_cis.chunk(2, axis=1)  # [B, 2, S, D]

    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    x_rotated = ops.stack([-x_imag, x_real], axis=-1).flatten(start_dim=3)

    out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    return out


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    dtype = query.dtype
    scale_factor = 1 / mint.sqrt(Tensor(query.shape[-1]))
    attn_weight = query @ key.swapaxes(-2, -1) * scale_factor.to(query.dtype)
    attn_weight = F.softmax(attn_weight.to(ms.float32), dim=-1).to(dtype)
    return attn_weight @ value


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: Tensor) -> Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = mint.cat([emb_h, emb_w], dim=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: Tensor) -> Tensor:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = mint.arange(embed_dim // 2, dtype=ms.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = ops.outer(pos, omega)  # (M, D/2), outer product

    emb_sin = mint.sin(out)  # (M, D/2)
    emb_cos = mint.cos(out)  # (M, D/2)

    emb = mint.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb


def get_3d_sincos_pos_embed(
    embed_dim: int,
    spatial_size: Union[int, Tuple[int, int]],
    temporal_size: int,
    spatial_interpolation_scale: float = 1.0,
    temporal_interpolation_scale: float = 1.0,
) -> Tensor:
    if embed_dim % 4 != 0:
        raise ValueError("`embed_dim` must be divisible by 4")
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)

    embed_dim_spatial = 3 * embed_dim // 4
    embed_dim_temporal = embed_dim // 4

    # 1. Spatial
    grid_h = mint.arange(spatial_size[1], dtype=ms.float32) / spatial_interpolation_scale
    grid_w = mint.arange(spatial_size[0], dtype=ms.float32) / spatial_interpolation_scale
    grid = ops.meshgrid(grid_w, grid_h)  # here w goes first
    grid = ops.stack(grid, axis=0)

    grid = grid.reshape([2, 1, spatial_size[1], spatial_size[0]])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    # 2. Temporal
    grid_t = mint.arange(temporal_size, dtype=ms.float32) / temporal_interpolation_scale
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t)

    # 3. Concat
    pos_embed_spatial = ops.unsqueeze(pos_embed_spatial, 0)
    pos_embed_spatial = mint.repeat_interleave(pos_embed_spatial, temporal_size, dim=0)  # [T, H*W, D // 4 * 3]

    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = mint.repeat_interleave(
        pos_embed_temporal, spatial_size[0] * spatial_size[1], dim=1
    )  # [T, H*W, D // 4]

    pos_embed = mint.cat([pos_embed_temporal, pos_embed_spatial], dim=-1)  # [T, H*W, D]
    return pos_embed


def get_timestep_embedding(
    timesteps: Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
) -> Tensor:
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -mint.log(Tensor(max_period)) * mint.arange(start=0, end=half_dim, dtype=ms.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = mint.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = mint.cat([mint.sin(emb), mint.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = mint.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class QKVAlltoALL(AlltoAll):
    def construct(
        self, q: Tensor, k: Tensor, v: Tensor, split_pad: Union[int, Tensor] = 0, concat_pad: Union[int, Tensor] = 0
    ) -> Tuple[Tensor, Tensor, Tensor]:
        x = mint.cat([q, k, v], dim=0)

        if split_pad > 0:
            padding = (len(x.shape) - self.split_dim - 1) * (0, 0) + (0, split_pad)
            x = mint.nn.functional.pad(x, padding)

        x = self.alltoall(x)

        if concat_pad > 0:
            x = x.narrow(self.concat_dim, 0, x.shape[self.concat_dim] - concat_pad)

        q, k, v = x.chunk(3, axis=0)
        return q, k, v

'''
class FP32LayerNorm(mint.nn.LayerNorm):
    def construct(self, input: Tensor) -> Tensor:
        dtype = input.dtype
        y = ops.layer_norm(
            input.to(ms.float32), self.normalized_shape, self.weight.to(ms.float32), self.bias.to(ms.float32), self.eps
        ).to(dtype)
        return y
'''

class FP32LayerNorm(nn.Cell):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine: bool = True, dtype=ms.float32):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(initializer("ones", normalized_shape, dtype=dtype))
            self.bias = Parameter(initializer("zeros", normalized_shape, dtype=dtype))
        else:
            self.weight = ops.ones(normalized_shape, dtype=dtype)
            self.bias = ops.zeros(normalized_shape, dtype=dtype)

    def construct(self, x: Tensor):
        normalized_shape = x.shape[-1:]
        # mint layer_norm fuses the operations in layer normorlization and it's faster than ops.LayerNorm
        ori_dtype = x.dtype
        x = mint.nn.functional.layer_norm(x.to(ms.float32), normalized_shape, self.weight.to(ms.float32), self.bias.to(ms.float32), self.eps).to(ori_dtype)

        return x



class ApproximateGELU(nn.Cell):
    def __init__(self, dim_in: int, dim_out: int, bias: bool = True, dtype: ms.Type = ms.float32) -> None:
        super().__init__()
        self.proj = mint.nn.Linear(dim_in, dim_out, bias=bias, dtype=dtype)

    def construct(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x * mint.sigmoid(1.702 * x)


class GELU(nn.Cell):
    def __init__(
        self, dim_in: int, dim_out: int, approximate: str = "none", bias: bool = True, dtype: ms.Type = ms.float32
    ) -> None:
        super().__init__()
        self.proj = mint.nn.Linear(dim_in, dim_out, bias=bias, dtype=dtype)
        self.approximate = approximate

    def gelu(self, gate: Tensor) -> Tensor:
        return F.gelu(gate, approximate=self.approximate)

    def construct(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class AdaLayerNorm(nn.Cell):
    def __init__(
        self,
        embedding_dim: int,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        output_dim = output_dim or embedding_dim * 2
        self.silu = nn.SiLU()
        self.linear = mint.nn.Linear(embedding_dim, output_dim, dtype=dtype)
        self.norm = FP32LayerNorm((output_dim // 2,), norm_eps, norm_elementwise_affine, dtype=dtype)

    def construct(self, x: Tensor, temb: Tensor) -> Tensor:
        temb = self.linear(self.silu(temb))
        shift, scale = temb.chunk(2, axis=1)
        shift = shift[:, None, :]
        scale = scale[:, None, :]
        x = self.norm(x) * (1 + scale) + shift
        return x


class CogVideoXLayerNormZero(nn.Cell):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = mint.nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias, dtype=dtype)
        self.norm = FP32LayerNorm((embedding_dim,), eps=eps, elementwise_affine=elementwise_affine, dtype=dtype)

    def construct(self, hidden_states: Tensor, encoder_hidden_states: Tensor, temb: Tensor) -> Tuple[Tensor, Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, axis=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]


class Attention(nn.Cell):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        qk_norm: Optional[str] = None,
        out_bias: bool = True,
        eps: float = 1e-5,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        inner_dim = dim_head * heads
        inner_kv_dim = inner_dim
        cross_attention_dim = query_dim
        out_dim = query_dim
        self.heads = heads

        if qk_norm is None:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        elif qk_norm == "layer_norm":
            self.norm_q = FP32LayerNorm((dim_head,), eps=eps, elementwise_affine=True, dtype=dtype)
            self.norm_k = FP32LayerNorm((dim_head,), eps=eps, elementwise_affine=True, dtype=dtype)
        else:
            raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None,'layer_norm'")

        self.to_q = mint.nn.Linear(query_dim, inner_dim, bias=bias, dtype=dtype)
        self.to_k = mint.nn.Linear(cross_attention_dim, inner_kv_dim, bias=bias, dtype=dtype)
        self.to_v = mint.nn.Linear(cross_attention_dim, inner_kv_dim, bias=bias, dtype=dtype)

        self.to_out = nn.SequentialCell(
            [mint.nn.Linear(inner_dim, out_dim, bias=out_bias, dtype=dtype), mint.nn.Dropout(p=dropout)]
        )

    def construct(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        image_rotary_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = mint.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, _, _ = encoder_hidden_states.shape

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)

        query = self.norm_q(query)
        key = self.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = scaled_dot_product_attention(query, key, value)

        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = self.to_out(hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.shape[1] - text_seq_length], axis=1
        )
        return hidden_states, encoder_hidden_states


class SequenceParallelAttention(nn.Cell):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        qk_norm: Optional[str] = None,
        out_bias: bool = True,
        eps: float = 1e-5,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        inner_dim = dim_head * heads
        inner_kv_dim = inner_dim
        cross_attention_dim = query_dim
        out_dim = query_dim
        self.heads = heads

        if qk_norm is None:
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        elif qk_norm == "layer_norm":
            self.norm_q = FP32LayerNorm((dim_head,), eps=eps, elementwise_affine=True, dtype=dtype)
            self.norm_k = FP32LayerNorm((dim_head,), eps=eps, elementwise_affine=True, dtype=dtype)
        else:
            raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None,'layer_norm'")

        self.to_q = mint.nn.Linear(query_dim, inner_dim, bias=bias, dtype=dtype)
        self.to_k = mint.nn.Linear(cross_attention_dim, inner_kv_dim, bias=bias, dtype=dtype)
        self.to_v = mint.nn.Linear(cross_attention_dim, inner_kv_dim, bias=bias, dtype=dtype)

        self.to_out = nn.SequentialCell(
            [mint.nn.Linear(inner_dim, out_dim, bias=out_bias, dtype=dtype), mint.nn.Dropout(p=dropout)]
        )

        self.sp_group = get_sequence_parallel_group()
        self.sp_size = get_group_size(self.sp_group)
        self.alltoall = QKVAlltoALL(split_dim=1, concat_dim=2, group=self.sp_group)
        self.alltoall_back = AlltoAll(split_dim=2, concat_dim=1, group=self.sp_group)

    def construct(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        image_rotary_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        encoder_query = self.to_q(encoder_hidden_states)
        encoder_key = self.to_k(encoder_hidden_states)
        encoder_value = self.to_v(encoder_hidden_states)

        batch_size, _, _ = encoder_hidden_states.shape

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)

        encoder_query = encoder_query.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)
        encoder_key = encoder_key.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)
        encoder_value = encoder_value.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)

        # b, h, sub_n, d -> b, sub_h, n, d
        query, key, value = self.alltoall(query, key, value)
        encoder_query, encoder_key, encoder_value = self.alltoall(encoder_query, encoder_key, encoder_value)

        text_seq_length = encoder_query.shape[2]

        query = mint.cat([encoder_query, query], dim=2)
        key = mint.cat([encoder_key, key], dim=2)
        value = mint.cat([encoder_value, value], dim=2)

        query = self.norm_q(query)
        key = self.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        hidden_states = scaled_dot_product_attention(query, key, value)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.shape[2] - text_seq_length], axis=2
        )
        # b, sub_h, n, d -> b, h, sub_n, d
        encoder_hidden_states = self.alltoall_back(encoder_hidden_states)
        hidden_states = self.alltoall_back(hidden_states)
        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        encoder_hidden_states = encoder_hidden_states.swapaxes(1, 2).reshape(batch_size, -1, self.heads * head_dim)

        hidden_states = self.to_out(hidden_states)
        encoder_hidden_states = self.to_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states


class FlashAttention(Attention):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        qk_norm: Optional[str] = None,
        out_bias: bool = True,
        eps: float = 1e-5,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__(
            query_dim=query_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            bias=bias,
            qk_norm=qk_norm,
            out_bias=out_bias,
            eps=eps,
            dtype=dtype,
        )
        self.flash_attention = FlashAttentionScore(heads, scale_value=dim_head**-0.5, input_layout="BSND")

    def construct(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        image_rotary_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        text_seq_length = encoder_hidden_states.shape[1]
        hidden_states = mint.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, _, _ = encoder_hidden_states.shape

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)

        query = self.norm_q(query)
        key = self.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # Reshape to the expected shape for Flash Attention
        query = mint.permute(query, (0, 2, 1, 3))
        key = mint.permute(key, (0, 2, 1, 3))
        value = mint.permute(value, (0, 2, 1, 3))

        _, _, _, hidden_states = self.flash_attention(query, key, value, None, None, None, None)

        hidden_states = hidden_states.reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = self.to_out(hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.shape[1] - text_seq_length], axis=1
        )
        return hidden_states, encoder_hidden_states


class SequenceParallelFlashAttention(SequenceParallelAttention):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        qk_norm: Optional[str] = None,
        out_bias: bool = True,
        eps: float = 1e-5,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__(
            query_dim=query_dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            bias=bias,
            qk_norm=qk_norm,
            out_bias=out_bias,
            eps=eps,
            dtype=dtype,
        )
        self.flash_attention = FlashAttentionScore(
            heads // self.sp_size, scale_value=dim_head**-0.5, input_layout="BSND"
        )

    def construct(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        image_rotary_emb: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        encoder_query = self.to_q(encoder_hidden_states)
        encoder_key = self.to_k(encoder_hidden_states)
        encoder_value = self.to_v(encoder_hidden_states)

        batch_size, _, _ = encoder_hidden_states.shape

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)

        encoder_query = encoder_query.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)
        encoder_key = encoder_key.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)
        encoder_value = encoder_value.view(batch_size, -1, self.heads, head_dim).swapaxes(1, 2)

        # b, h, sub_n, d -> b, sub_h, n, d
        query, key, value = self.alltoall(query, key, value)
        encoder_query, encoder_key, encoder_value = self.alltoall(encoder_query, encoder_key, encoder_value)

        text_seq_length = encoder_query.shape[2]

        query = mint.cat([encoder_query, query], dim=2)
        key = mint.cat([encoder_key, key], dim=2)
        value = mint.cat([encoder_value, value], dim=2)

        query = self.norm_q(query)
        key = self.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(query[:, :, text_seq_length:], image_rotary_emb)
            key[:, :, text_seq_length:] = apply_rotary_emb(key[:, :, text_seq_length:], image_rotary_emb)

        # Reshape to the expected shape for Flash Attention
        query = mint.permute(query, (0, 2, 1, 3))
        key = mint.permute(key, (0, 2, 1, 3))
        value = mint.permute(value, (0, 2, 1, 3))

        _, _, _, hidden_states = self.flash_attention(query, key, value, None, None, None, None)

        # Reshape back for sequence parallel
        hidden_states = mint.permute(hidden_states, (0, 2, 1, 3))

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, hidden_states.shape[2] - text_seq_length], axis=2
        )
        # b, sub_h, n, d -> b, h, sub_n, d
        encoder_hidden_states = self.alltoall_back(encoder_hidden_states)
        hidden_states = self.alltoall_back(hidden_states)

        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        encoder_hidden_states = encoder_hidden_states.swapaxes(1, 2).reshape(batch_size, -1, self.heads * head_dim)

        hidden_states = self.to_out(hidden_states)
        encoder_hidden_states = self.to_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states


class FeedForward(nn.Cell):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu-approximate",
        final_dropout: bool = False,
        inner_dim: Optional[int] = None,
        bias: bool = True,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias, dtype=dtype)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias, dtype=dtype)
        else:
            raise ValueError(f"Unknown activation function `{activation_fn}`.")

        self.net = nn.SequentialCell(
            [act_fn, mint.nn.Dropout(p=dropout), mint.nn.Linear(inner_dim, dim_out, bias=bias, dtype=dtype)]
        )
        if final_dropout:
            self.net.append(mint.nn.Dropout(dropout))

    def construct(self, hidden_states: Tensor) -> Tensor:
        return self.net(hidden_states)


class CogVideoXPatchEmbed(nn.Cell):
    def __init__(
        self,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        in_channels: int = 16,
        embed_dim: int = 1920,
        text_embed_dim: int = 4096,
        bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_positional_embeddings: bool = True,
        use_learned_positional_embeddings: bool = True,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_frames = sample_frames
        self.temporal_compression_ratio = temporal_compression_ratio
        self.max_text_seq_length = max_text_seq_length
        self.spatial_interpolation_scale = spatial_interpolation_scale
        self.temporal_interpolation_scale = temporal_interpolation_scale
        self.use_positional_embeddings = use_positional_embeddings
        self.use_learned_positional_embeddings = use_learned_positional_embeddings

        if self.patch_size[0] == 1:
            self.proj = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=(patch_size[1], patch_size[2]),
                stride=(patch_size[1], patch_size[2]),
                pad_mode="pad",
                has_bias=bias,
                dtype=dtype,
            )
        else:
            self.proj = mint.nn.Linear(in_channels * np.prod(patch_size).item(), embed_dim, dtype=dtype)

        self.text_proj = mint.nn.Linear(text_embed_dim, embed_dim, dtype=dtype)

        if use_positional_embeddings or use_learned_positional_embeddings:
            pos_embedding = self._get_positional_embeddings(sample_height, sample_width, sample_frames)
            if use_learned_positional_embeddings:
                self.pos_embedding = Parameter(pos_embedding, requires_grad=True)
            else:
                self.pos_embedding = pos_embedding

    def _get_positional_embeddings(self, sample_height: int, sample_width: int, sample_frames: int) -> Tensor:
        post_patch_height = sample_height // self.patch_size[1]
        post_patch_width = sample_width // self.patch_size[2]
        post_time_compression_frames = (sample_frames - 1) // self.temporal_compression_ratio + 1
        num_patches = post_patch_height * post_patch_width * post_time_compression_frames

        pos_embedding = get_3d_sincos_pos_embed(
            self.embed_dim,
            (post_patch_width, post_patch_height),
            post_time_compression_frames,
            self.spatial_interpolation_scale,
            self.temporal_interpolation_scale,
        )
        pos_embedding = pos_embedding.flatten(start_dim=0, end_dim=1)
        joint_pos_embedding = mint.zeros((1, self.max_text_seq_length + num_patches, self.embed_dim))
        joint_pos_embedding[:, self.max_text_seq_length :] = pos_embedding
        return joint_pos_embedding

    def construct(self, text_embeds: Tensor, image_embeds: Tensor) -> Tensor:
        text_embeds = self.text_proj(text_embeds)

        batch_size, num_frames, channels, height, width = image_embeds.shape

        if self.patch_size[0] == 1:
            image_embeds = image_embeds.reshape(-1, channels, height, width)
            image_embeds = self.proj(image_embeds)
            image_embeds = image_embeds.view(batch_size, num_frames, *image_embeds.shape[1:])
            image_embeds = image_embeds.flatten(start_dim=3).swapaxes(
                2, 3
            )  # [batch, num_frames, height x width, channels]
            image_embeds = image_embeds.flatten(
                start_dim=1, end_dim=2
            )  # [batch, num_frames x height x width, channels]
        else:
            assert num_frames % self.patch_size[0] == 0
            image_embeds = image_embeds.permute(0, 1, 3, 4, 2)
            image_embeds = image_embeds.reshape(
                batch_size,
                num_frames // self.patch_size[0],
                self.patch_size[0],
                height // self.patch_size[1],
                self.patch_size[1],
                width // self.patch_size[2],
                self.patch_size[2],
                channels,
            )
            image_embeds = (
                image_embeds.permute(0, 1, 3, 5, 7, 2, 4, 6)
                .flatten(start_dim=4, end_dim=7)
                .flatten(start_dim=1, end_dim=3)
            )
            image_embeds = self.proj(image_embeds)

        embeds = mint.cat(
            [text_embeds, image_embeds], dim=1
        ).contiguous()  # [batch, seq_length + num_frames x height x width, channels]

        if self.use_positional_embeddings or self.use_learned_positional_embeddings:
            if self.use_learned_positional_embeddings and (self.sample_width != width or self.sample_height != height):
                raise ValueError(
                    "It is currently not possible to generate videos at a different resolution that the defaults. "
                    "This should only be the case with 'THUDM/CogVideoX-5b-I2V'."
                    "If you think this is incorrect, please open an issue at https://github.com/huggingface/diffusers/issues."
                )

            pre_time_compression_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

            if (
                self.sample_height != height
                or self.sample_width != width
                or self.sample_frames != pre_time_compression_frames
            ):
                pos_embedding = self._get_positional_embeddings(height, width, pre_time_compression_frames)
                pos_embedding = pos_embedding.to(embeds.dtype)
            else:
                pos_embedding = self.pos_embedding.to(embeds.dtype)

            embeds = embeds + pos_embedding

        return embeds


class Timesteps(nn.Cell):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float, scale: int = 1) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.scale = scale

    def construct(self, timesteps: Tensor) -> Tensor:
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            scale=self.scale,
        )
        return t_emb


class TimestepEmbedding(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        sample_proj_bias: bool = True,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        self.linear_1 = mint.nn.Linear(in_channels, time_embed_dim, sample_proj_bias, dtype=dtype)

        if act_fn == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation function `{act_fn}`.")

        time_embed_dim_out = time_embed_dim
        self.linear_2 = mint.nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias, dtype=dtype)

    def construct(self, sample: Tensor) -> Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)
        return sample


class CogVideoXBlock(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        enable_flash_attention: bool = False,
        enable_sequence_parallelism: bool = False,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True, dtype=dtype
        )

        if enable_sequence_parallelism:
            attn_ = SequenceParallelFlashAttention if enable_flash_attention else SequenceParallelAttention
        else:
            attn_ = FlashAttention if enable_flash_attention else Attention

        self.attn1 = attn_(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            dtype=dtype,
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(
            time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True, dtype=dtype
        )

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
            dtype=dtype,
        )

    def construct(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb: Tensor,
        image_rotary_emb: Optional[Tensor] = None,
    ) -> Tensor:
        text_seq_length = encoder_hidden_states.shape[1]

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = mint.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states


class CogVideoXTransformer3DModel(nn.Cell):
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: Optional[int] = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
        enable_flash_attention: bool = False,
        enable_sequence_parallelism: bool = False,
        use_recompute: bool = False,
        rope_grid_type: Literal["linspace", "slice"] = "linspace",
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.use_rotary_positional_embeddings = use_rotary_positional_embeddings
        self.patch_size = patch_size
        self.sample_height = sample_height
        self.sample_width = sample_width
        self._dtype = dtype
        self.attention_head_dim = attention_head_dim
        self.hidden_size = self.attention_head_dim * num_attention_heads
        self.num_heads = num_attention_heads
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.rope_grid_type = rope_grid_type

        inner_dim = num_attention_heads * attention_head_dim

        if not self.use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not self.use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
            dtype=dtype,
        )
        self.embedding_dropout = mint.nn.Dropout(dropout)

        # 2. Time embeddings
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn, dtype=dtype)

        # 3. Define spatio-temporal transformers blocks
        self.transformer_blocks = nn.CellList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    enable_flash_attention=enable_flash_attention,
                    enable_sequence_parallelism=enable_sequence_parallelism,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = FP32LayerNorm((inner_dim,), norm_eps, norm_elementwise_affine, dtype=dtype)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            dtype=dtype,
        )
        self.proj_out = mint.nn.Linear(inner_dim, np.prod(patch_size).item() * out_channels, dtype=dtype)

        if self.enable_sequence_parallelism:
            sp_group = get_sequence_parallel_group()
            logger.info(f"Initialize STDIT-v3 model with sequence parallel group `{sp_group}`.")
            self.sp_size = get_group_size(sp_group)
            assert self.num_heads % self.sp_size == 0
            self.split_forward_gather_backward = SplitFowardGatherBackward(dim=1, grad_scale="down", group=sp_group)
            self.gather_forward_split_backward = GatherFowardSplitBackward(dim=1, grad_scale="up", group=sp_group)

        if use_recompute:
            for block in self.transformer_blocks:
                block.recompute()

    @property
    def dtype(self):
        return self._dtype

    def construct(
        self, x: Tensor, timestep: Tensor, y: Tensor, image_rotary_emb: Optional[Tensor] = None, **kwargs
    ) -> Tensor:
        print('x shape', x.shape)

        hidden_states = ops.transpose(x, (0, 2, 1, 3, 4)).to(self.dtype)  # n, t, c, h, w
        encoder_hidden_states = y.to(self.dtype)

        batch_size, num_frames, _, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb)

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        if self.enable_sequence_parallelism:
            assert hidden_states.shape[1] % self.sp_size == 0
            assert encoder_hidden_states.shape[1] % self.sp_size == 0
            hidden_states = self.split_forward_gather_backward(hidden_states)
            encoder_hidden_states = self.split_forward_gather_backward(encoder_hidden_states)

        # 3. Transformer blocks
        for block in self.transformer_blocks:
            hidden_states, encoder_hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
            )

        if self.enable_sequence_parallelism:
            hidden_states = self.gather_forward_split_backward(hidden_states)
            encoder_hidden_states = self.gather_forward_split_backward(encoder_hidden_states)

        if not self.use_rotary_positional_embeddings:
            # CogVideoX-2B
            hidden_states = self.norm_final(hidden_states)
        else:
            # CogVideoX-5B
            hidden_states = mint.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        if self.patch_size[0] == 1:
            output = hidden_states.reshape(
                batch_size,
                num_frames,
                height // self.patch_size[1],
                width // self.patch_size[2],
                -1,
                self.patch_size[1],
                self.patch_size[2],
            )
            output = (
                output.permute(0, 4, 1, 2, 5, 3, 6).flatten(start_dim=5, end_dim=6).flatten(start_dim=3, end_dim=4)
            )  # n, c, t, h, w
        else:
            output = hidden_states.reshape(
                batch_size,
                (num_frames + self.patch_size[0] - 1) // self.patch_size[0],
                height // self.patch_size[1],
                width // self.patch_size[2],
                -1,
                self.patch_size[0],
                self.patch_size[1],
                self.patch_size[2],
            )
            output = (
                output.permute(0, 4, 1, 5, 2, 6, 3, 7)
                .flatten(start_dim=6, end_dim=7)
                .flatten(start_dim=4, end_dim=5)
                .flatten(start_dim=2, end_dim=3)
            )  # n, c, t, h, w

        return output

    def construct_with_cfg(
        self,
        x: Tensor,
        timestep: Tensor,
        y: Tensor,
        image_rotary_emb: Optional[Tensor] = None,
        cfg_scale: Union[float, Tensor] = 6.0,
        **kwargs,
    ) -> Tensor:
        x = mint.chunk(x, 2, 0)[0]
        x = mint.tile(x, (2, 1, 1, 1, 1))
        x = self.construct(x, timestep, y, image_rotary_emb=image_rotary_emb, **kwargs)
        pred_cond, pred_uncond = mint.chunk(x, 2, 0)
        pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        pred = mint.tile(pred, (2, 1, 1, 1, 1))
        return pred

    def load_from_checkpoint(self, ckpt_path: List[str]) -> None:
        assert isinstance(ckpt_path, list)
        format = "safetensors" if ckpt_path[0].endswith(".safetensors") else "ckpt"

        if format == "ckpt":
            if len(ckpt_path) > 1:
                raise ValueError("It can read weight from single file (.ckpt) only.")
            parameter_dict = ms.load_checkpoint(ckpt_path[0])
            parameter_dict = {k.replace("network.", ""): v for k, v in parameter_dict.items()}
        else:
            parameter_dict = dict()
            for path in ckpt_path:
                with safe_open(path, framework="np") as f:
                    for k in f.keys():
                        parameter_dict[k] = Parameter(Tensor(f.get_tensor(k), dtype=self.dtype))

        param_not_load, ckpt_not_load = ms.load_param_into_net(self, parameter_dict, strict_load=True)
        assert len(param_not_load) == 0 and len(ckpt_not_load) == 0


def CogVideoX_2B(from_pretrained: Optional[str] = None, **kwargs) -> CogVideoXTransformer3DModel:
    model = CogVideoXTransformer3DModel(
        num_attention_heads=30, num_layers=30, use_rotary_positional_embeddings=False, **kwargs
    )

    if from_pretrained is not None:
        model.load_from_checkpoint(from_pretrained)
    return model


def CogVideoX_5B(from_pretrained: Optional[str] = None, **kwargs) -> CogVideoXTransformer3DModel:
    # FIXME: debugging
    model = CogVideoXTransformer3DModel(num_attention_heads=48, num_layers=42, use_rotary_positional_embeddings=True, **kwargs)
    # model = CogVideoXTransformer3DModel(num_attention_heads=12, num_layers=42, use_rotary_positional_embeddings=True, **kwargs)

    if from_pretrained is not None:
        model.load_from_checkpoint(from_pretrained)
    return model


def CogVideoX_5B_v1_5(from_pretrained: Optional[str] = None, **kwargs) -> CogVideoXTransformer3DModel:
    model = CogVideoXTransformer3DModel(
        num_attention_heads=48,
        num_layers=42,
        use_rotary_positional_embeddings=True,
        sample_width=300,
        sample_height=300,
        sample_frames=81,
        patch_size=(2, 2, 2),
        patch_bias=False,
        rope_grid_type="slice",
        **kwargs,
    )

    if from_pretrained is not None:
        model.load_from_checkpoint(from_pretrained)
    return model
