import math
import numpy as np
import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import Normal, XavierUniform, initializer
from ..utils.helpers import to_2tuple

class PatchEmbed(nn.Cell):
    def __init__(
        self,
        patch_size=2,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        # print('D--: patch_size, ', patch_size)

        # TODO: doing here. replace with conv2d. refer to opensora
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            has_bias=bias,
            pad_mode='valid',
            bias_init='zeros',
        )
        # nn.init.xavier_uniform_(self.proj.weight.view(self.proj.weight.size(0), -1))
        w = self.proj.weight
        w_flatted = w.reshape(w.shape[0], -1)
        w.set_data(initializer(XavierUniform(), w_flatted.shape, w_flatted.dtype).reshape(w.shape))
        # nn.init.zeros_(self.proj.bias)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def construct(self, x):
        x = self.proj(x)
        if self.flatten:
            # (B C T H W) -> (B C THW) -> (B THW C)
            x = x.flatten(start_dim=2).transpose((0, 2, 1))  # BCHW -> BNC
        x = self.norm(x)
        return x

class TextProjection(nn.Cell):
    """
    Projects text embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_channels, hidden_size, act_layer, dtype=None):
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.linear_1 = nn.Dense(
            in_channels,
            hidden_size,
            has_bias=True,
        )
        self.act_1 = act_layer()
        self.linear_2 = nn.Dense(
            hidden_size,
            hidden_size,
            has_bias=True,
        )

    def construct(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class SinusoidalEmbedding(nn.Cell):
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        half = dim // 2
        self._freqs = ms.Tensor(
            np.expand_dims(
                np.exp(-math.log(max_period) * np.arange(start=0, stop=half, dtype=np.float32) / half), axis=0
            )
        )
        self._dim = dim

    def construct(self, t):
        args = t[:, None] * self._freqs
        embedding = ops.cat([ops.cos(args), ops.sin(args)], axis=-1)
        if self._dim % 2:
            embedding = ops.cat([embedding, ops.zeros_like(embedding[:, :1])], axis=-1)
        return embedding


def init_normal(param, mean=0., std=1.) -> None:
    param.set_data(initializer(Normal(std, mean), param.shape, param.dtype))


class TimestepEmbedder(nn.Cell):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size,
        act_layer,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.SequentialCell(
            nn.Dense(
                frequency_embedding_size, hidden_size, has_bias=True,
            ),
            act_layer(),
            nn.Dense(hidden_size, out_size, has_bias=True),
        )
        init_normal(self.mlp[0].weight, std=0.02)
        init_normal(self.mlp[2].weight, std=0.02)

        self.timestep_embedding = SinusoidalEmbedding(frequency_embedding_size, max_period=max_period)

    def construct(self, t):
        t_freq = self.timestep_embedding(t) # .to(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb
