# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Final, List, Literal, Optional, Sequence, Set, Tuple, Type, Union
import math

import numpy as np
from safetensors import safe_open

import mindspore as ms
from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore.mint.nn import LayerNorm
#from mindspore import amp
from mindone.utils.amp import auto_mixed_precision

from mindone.transformers.mindspore_adapter.attention import scaled_dot_product_attention

from .utils import set_model_param_dtype
from .timm import (
    QuickGELUActivation,
    AttentionPoolLatent,
    DropPath,
    LayerType,
    Mlp,
    PatchDropout,
    PatchEmbed,
    no_grad_trunc_normal_,
    resample_abs_pos_embed,
)

HACK_DEBUG = False

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""The original timm.models.layers.weight_init.trunc_normal_ can not handle bfloat16 yet, here we first
    convert the tensor to float32, apply the trunc_normal_() in float32, and then convert it back to its original dtype.
    Fills the input Tensor with values drawn from a truncated normal distribution. The values are effectively drawn
    from the normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = mint.empty(3, 5)
        >>> trunc_normal_(w)
    """

    # with ms._no_grad(): # dosn't support graph mode
    dtype = tensor.dtype
    tensor_fp32 = tensor.float()
    tensor_fp32 = no_grad_trunc_normal_(tensor_fp32, mean, std, a, b)
    tensor_dtype = tensor_fp32.to(dtype=dtype)
    tensor.copy_(tensor_dtype)

    ops.stop_gradient(tensor)


def init_weights(self):
    if self.pos_embed is not None:
        trunc_normal_(self.pos_embed, std=self.pos_embed.shape[1] ** -0.5)
    trunc_normal_(self.latent, std=self.latent_dim**-0.5)


def init_weights_vit_timm(module: nn.Cell, name: str = "") -> None:
    """ViT weight initialization, original timm impl (for reproducibility)"""
    if isinstance(module, mint.nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            module.bias.fill_(0)
    elif hasattr(module, "init_weights"):
        module.init_weights()


from mindformers.modules.layers import Linear as MF_Linear

class Attention(nn.Cell):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Cell = LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        # self.fused_attn = use_fused_attn()
        self.fused_attn = True
        
        # default in timm: qqqq kkkk vvv 
        # self.qkv = mint.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)  # better precision than mint Linear
        # self.qkv = mint.nn.functional.linear

        
        # TODO: try to change re-arrange map? 
        # self.qkv.weight = nn.Parameter(self.qkv.weight.reshape(3, 16, 72, 1152).permute(1, 0, 2, 3).reshape(-1, 1152)).to(target_device, dtype=target_dtype)
        # self.qkv.bias = nn.Parameter(self.qkv.bias.reshape(3, 16, 72).permute(1, 0, 2).reshape(-1)).to(target_device, dtype=target_dtype)

        self.q_norm = norm_layer([self.head_dim]) if qk_norm else nn.Identity()
        self.k_norm = norm_layer([self.head_dim]) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(p=attn_drop)
        # self.proj = mint.nn.Linear(dim, dim)
        self.proj = nn.Dense(dim, dim)

        self.proj_drop = nn.Dropout(p=proj_drop) if proj_drop > 0.0 else nn.Identity()

        self.use_fa = True

    def construct(self, x: Tensor) -> Tensor:
        # N - seq len
        B, N, C = x.shape
        # DEBUG texthawk: num_heads 16, head_dim 72 
        # Linear projection
        if HACK_DEBUG:
            print("D--: qkv projection input error: ")
            from compare import print_diff; diff, pta_val = print_diff(x.asnumpy().transpose(1,0,2), "/home/hyx/models/texthawk_vision/features_self_attn/module_siglip_block_0_layer_0_self-attn_before_qkv.pkl")

        # (B S H) -> (B S H*3), qqkkvv
        qkv = self.qkv(x)
        # qkv = mint.matmul(x, self.qkv.weight.transpose(1, 0)) + self.qkv.bias
        
        if not self.use_fa: 
            # (B S H*3) -> (B S 3 num_head head_dim) -> (3 B num_head S head_dim) = (3 B 16 1024 72)  # => BNSH format 
            qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)  # (B num_heads S head_dim)
            
            # train infer save merged_qkv:  (S B H*3), qkvqkv 
            # train infer save query:  (S B num_heads head_dim), 4K: (1024, 5, 16, 72)
            if HACK_DEBUG:
                print("D--: q k v error after qkv projection: ")
                from compare import print_diff; diff, pta_val = print_diff(q.asnumpy().transpose(2,0,1,3), "/home/hyx/models/texthawk_vision/features_self_attn/module_siglip_block_0_layer_0_self-attn_query.pkl")
                from compare import print_diff; diff, pta_val = print_diff(k.asnumpy().transpose(2,0,1,3), "/home/hyx/models/texthawk_vision/features_self_attn/module_siglip_block_0_layer_0_self-attn_key.pkl")
                from compare import print_diff; diff, pta_val = print_diff(v.asnumpy().transpose(2,0,1,3), "/home/hyx/models/texthawk_vision/features_self_attn/module_siglip_block_0_layer_0_self-attn_value.pkl")
                # import pdb; pdb.set_trace()
        else:
            # (B S 3H) -> (S B 3H) ->  (S B 3 H) -> 3 (S B H)
            q, k, v = qkv.transpose((1, 0, 2)).reshape(N, B, 3, self.num_heads * self.head_dim).unbind(2)

            if HACK_DEBUG:
                print("D--: q k v error after qkv projection: ")
                from compare import print_diff; diff, pta_val = print_diff(q.asnumpy().reshape(N, B, self.num_heads, self.head_dim), "/home/hyx/models/texthawk_vision/features_self_attn/module_siglip_block_0_layer_0_self-attn_query.pkl")

        q, k = self.q_norm(q), self.k_norm(k)  # identity
            
        # in training infer: (S B num_heads head_dim) -> (SBH) format, and use torch_npu.npu_fusion_attention
        #  ==> 4K: q: (1024 5 1152) 
        #  profile 8K data: (1024 33 144) ==> num_heads 16 -> 2
        
        # here: BNSD format
        # TODO: use SBH format and ms flash attn score API to compute it
        # refer to: mindspeed_mm\attention_patches\dot_product_attention_qwen2vl.py
        if self.use_fa: 
            x = ms.ops.flash_attention_score(
                q, k, v, self.num_heads, input_layout='SBH', scalar_value=1.0 / math.sqrt(self.head_dim),
                )
            x = x.transpose(1, 0, 2) # SBH -> BSH
        else:
            if self.fused_attn:
                x = scaled_dot_product_attention(
                    q,
                    k,
                    v,
                )
            else:
                q = q * self.scale
                # attn = ops.bmm(q, k.transpose(0, 1, 3, 2))
                attn = q @ k.transpose(0, 1, 3, 2)

                attn = attn.to(ms.float32)
                attn = mint.softmax(attn, dim=-1).to(q.dtype)

                attn = self.attn_drop(attn)
                # x = ops.bmm(attn, v)
                x = attn @ v
            # B N S D -> B S N D -> B S H
            x = x.transpose(0, 2, 1, 3).reshape(B, N, C)

        if HACK_DEBUG:
            print("D--: x error after dot product attn: ")
            from compare import print_diff; diff, pta_val = print_diff(x.asnumpy().transpose(1,0,2), "/home/hyx/models/texthawk_vision/features_self_attn/module_siglip_block_0_layer_0_self-attn_attn_output.pkl")
            import pdb; pdb.set_trace()

        x = self.proj(x)
        # x = mint.matmul(x, self.proj.weight.transpose(1, 0))
        # x = x + self.proj.bias

        x = self.proj_drop(x)
        return x


class LayerScale(nn.Cell):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = Parameter(init_values * mint.ones(dim))

    def construct(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Cell = mint.nn.GELU,
        norm_layer: nn.Cell = LayerNorm,
        mlp_layer: nn.Cell = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer([dim])
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer([dim])
        # FIXME: Temp fix: texthawk use QuickGELU for MLP act
        # print("D--: act layer: ", act_layer)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.cur_layer = -1

    def construct(self, x: Tensor) -> Tensor:
        if HACK_DEBUG: 
            import pdb; pdb.set_trace()
            from compare import read_pickle_value, print_diff
            print("D--: Block input error: ")
            print_diff(x.asnumpy().transpose(1,0,2), f"/home/hyx/models/texthawk_vision/features_full/texthawk_ds_features_gt_full/module_siglip_block_0_layer_0_before_input_layernorm.pkl") 

        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        h = self.norm1(x)
        # after_input_layernorm.pkl mre 0
        
        h = self.attn(h)
        h = self.ls1(h)
        x = x + self.drop_path1(h)

        if HACK_DEBUG: 
            # import pdb; pdb.set_trace() 
            print("D--: x error after out projction + bias + residual: ")
            print_diff(x.asnumpy().transpose(1,0,2), "/home/hyx/models/texthawk_vision/features_full/texthawk_ds_features_gt_full/module_siglip_block_0_layer_0_before_pre_mlp_layernorm.pkl") 
            import pdb; pdb.set_trace()

        # before_pre_mlp_layernorm.pkl, mre 0.001
        h = self.norm2(x)
        # after_pre_mlp_layernorm.pkl, mre 0.019
        
        # DEBUGGING
        '''
        import pdb; pdb.set_trace()
        force_input = "/home/hyx/models/texthawk_vision/features_full/texthawk_ds_features_gt_full/module_siglip_block_0_layer_0_after_pre_mlp_layernorm.pkl" 
        from compare import read_pickle_value, print_diff
        h = ms.Tensor(read_pickle_value(force_input).transpose(1,0,2))
        print_diff(h.asnumpy().transpose(1,0,2), force_input)

        force_input_x = "/home/hyx/models/texthawk_vision/features_full/texthawk_ds_features_gt_full/module_siglip_block_0_layer_0_before_pre_mlp_layernorm.pkl" 
        x = ms.Tensor(read_pickle_value(force_input_x).transpose(1,0,2))
        print_diff(x.asnumpy().transpose(1,0,2), force_input_x)
        print("x and h are set to the same as training input")
        '''

        h = self.mlp(h)
        

        h = self.ls2(h)
        x = x + self.drop_path2(h)

        # before_return.pkl, mre 0.037

        if HACK_DEBUG: 
            print("D--: x error after norm, mlp, residual ")
            ref_output = "/home/hyx/models/texthawk_vision/features_full/texthawk_ds_features_gt_full/module_siglip_block_0_layer_0_before_return.pkl" 
            print_diff(x.asnumpy().transpose(1,0,2), ref_output)
            import pdb; pdb.set_trace()

        return x


class VisionTransformer(nn.Cell):
    """Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    dynamic_img_size: Final[bool]

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[nn.Cell] = Block,
        mlp_layer: Type[nn.Cell] = Mlp,
        ignore_head: bool = False,
    ) -> None:
        """
        Args:
            img_size: Input image size.
            patch_size: Patch size.
            in_chans: Number of image input channels.
            num_classes: Mumber of classes for classification head.
            global_pool: Type of global pooling for final sequence (default: 'token').
            embed_dim: Transformer embedding dimension.
            depth: Depth of transformer.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of mlp hidden dim to embedding dim.
            qkv_bias: Enable bias for qkv projections if True.
            init_values: Layer-scale init values (layer-scale enabled if not None).
            class_token: Use class token.
            no_embed_class: Don't include position embeddings for class (or reg) tokens.
            reg_tokens: Number of register tokens.
            fc_norm: Pre head norm after pool (instead of before), if None, enabled when global_pool == 'avg'.
            drop_rate: Head dropout rate.
            pos_drop_rate: Position embedding dropout rate.
            attn_drop_rate: Attention dropout rate.
            drop_path_rate: Stochastic depth rate.
            weight_init: Weight initialization scheme.
            embed_layer: Patch embedding layer.
            norm_layer: Normalization layer.
            act_layer: MLP activation layer.
            block_fn: Transformer block layer.
        """
        super().__init__()
        assert global_pool in ("", "avg", "token", "map")
        assert class_token or global_pool != "token"
        use_fc_norm = global_pool == "avg" if fc_norm is None else fc_norm

        norm_layer = partial(LayerNorm, eps=1e-6)

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.num_prefix_tokens += reg_tokens
        self.num_reg_tokens = reg_tokens
        self.has_class_token = class_token
        self.no_embed_class = no_embed_class  # don't embed prefix positions (includes reg)
        self.dynamic_img_size = dynamic_img_size
        self.grad_checkpointing = False
        self.ignore_head = ignore_head
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        embed_args = {}
        if dynamic_img_size:
            # flatten deferred until after pos embed
            embed_args.update(dict(strict_img_size=False, output_fmt="NHWC"))
        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
            dynamic_img_pad=dynamic_img_pad,
            **embed_args,
        )
        num_patches = self.patch_embed.num_patches
        self.seq_length = num_patches


        self.cls_token = Parameter(mint.zeros(1, 1, embed_dim)) if class_token else None
        self.reg_token = Parameter(mint.zeros(1, reg_tokens, embed_dim)) if reg_tokens else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = Parameter(
            ms.Tensor(np.random.normal(size=(1, embed_len, embed_dim)).astype(np.float32) * 0.02)
        )
        # self.pos_embed = mint.nn.Embedding(embed_len, embed_dim, dtype=ms.float32)

        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=self.num_prefix_tokens,
            )
        else:
            self.patch_drop = nn.Identity()
        self.norm_pre = norm_layer([embed_dim]) if pre_norm else nn.Identity()

        dpr = [x.item() for x in mint.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.SequentialCell(
            *[
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    init_values=init_values,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer([embed_dim]) if not use_fc_norm else nn.Identity()

        # Classifier Head
        # DEBUG: not used in texthawk
        if global_pool == "map":
            AttentionPoolLatent.init_weights = init_weights
            self.attn_pool = AttentionPoolLatent(
                self.embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        else:
            self.attn_pool = None
        self.fc_norm = norm_layer([embed_dim]) if use_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(p=drop_rate)
        self.head = mint.nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()


    def load_from_checkpoint(self, ckpt_path, add_prefix="", amp_level=None):
        # mainly used in unit test
        parameter_dict = dict()
        if ckpt_path.endswith(".bin"):
            import torch

            sd = torch.load(ckpt_path, weights_only=True)
            # filter to keep vision_tower params only and remove prefix vision_model.vision_tower
            pnames = [p for p in sd]
            for p in pnames:
                if "vision_tower" not in p:
                    sd.pop(p)
                else:
                    # remove prefix
                    new_pname = p.replace("vision_model.vision_tower.", "")
                    # special: weight (pt) - > embedding_table (ms)
                    if "embedding.weight" in p:
                        new_pname = new_pname.replace("embedding.weight", "embedding.embedding_table")
                    # elif "norm" in p:
                    #    new_pname = new_pname.replace("weight", "gamma").replace("bias", "beta")

                    sd[new_pname] = sd.pop(p)

            # get net param dtype
            param_dtype = tuple(self.get_parameters())[0].dtype
            print("Get siglip param dtype: ", param_dtype)

            for pname in sd:
                # print(pname, sd[pname].shape, sd[pname].dtype)
                np_val = sd[pname].cpu().detach().float().numpy()
                # TODO: support bf16 param loading
                parameter_dict[pname] = ms.Parameter(ms.Tensor(np_val, dtype=param_dtype))

        elif ckpt_path.endswith(".ckpt"):
            parameter_dict = ms.load_checkpoint(ckpt_path)
        elif ckpt_path.endswith(".safetensors"):
            parameter_dict = ms.load_checkpoint(ckpt_path, format='safetensors')
            pnames = list(parameter_dict.keys())
            is_from_texthawk = pnames[0].startswith("vision_model.")

            if is_from_texthawk:
                print("D--: load param from texthawk")
                prefix = "vision_model.encoder.vit." 
                for p in pnames:
                    if prefix not in p:
                        parameter_dict.pop(p)
                    else: 
                        # name refine
                        new_pname = p.replace(prefix, "")
                        # if "vit.pos_embed.weight" in p:
                        #     new_pname = new_pname.replace(".weight", "")
                        
                        # FIXME: due to we change patch_embed externally, patch_embed weight name is 'pos_embed.weight' rather than 'vit.pos_embed.weight'
                        if not "vit.pos_embed.weight" in p:
                            new_pname = add_prefix + new_pname
                        # FIXME: somehow, using auto_mixed_precision to set LayerNorm in fp32 will add _backbone to the param name...
                        if "norm" in new_pname and amp_level is not None:
                            new_pname = new_pname.replace(".bias", "._backbone.bias")
                            new_pname = new_pname.replace(".weight", "._backbone.weight")

                        # value shaping
                        weight  = parameter_dict.pop(p)
                        if "vit.patch_embed.proj.weight" in p:
                            # in conversion script: value = value.permute(0, 2, 3, 1).flatten(1).clone()
                            # torch conv weight (cout, cin, h, w) -> (cout, h, w, cin) -> (cout, h*w*cin)
                            # revert: value.reshape(cout, h, w, cin).permute(0, 3, 1, 2)
                            weight = ops.reshape(weight, (self.embed_dim, self.patch_size, self.patch_size, 3))  #
                            weight = ops.transpose(weight, (0, 3, 1, 2))
                            parameter_dict[new_pname] = ms.Parameter(weight, name=new_pname)
                            print("D--: ", new_pname, parameter_dict[new_pname].dtype)
                            print(parameter_dict[new_pname] )
                        else:
                            parameter_dict[new_pname] = weight

            else:
                # checkpoint from timm/ViT-SO400M-14-SigLIP-384
                for p in pnames:
                    if "visual." not in p:
                        # exclude text transformers
                        parameter_dict.pop(p)
                    else:
                        new_pname = p.replace("visual.trunk.", "")
                        parameter_dict[new_pname] = parameter_dict.pop(p)
        else: 
            raise ValueError("Unsupported checkpoint format")

        param_not_load, ckpt_not_load = ms.load_param_into_net(self, parameter_dict, strict_load=True)
        if param_not_load:
            print(
                "Net params not load: {}, Total net params not loaded: {}".format(param_not_load, len(param_not_load))
            )
        if ckpt_not_load:
            print(
                "Ckpt params not load: {}, Total ckpt params not loaded: {}".format(ckpt_not_load, len(ckpt_not_load))
            )
        assert len(ckpt_not_load) == 0, f"These vision parameters from checkpoint are NOT loaded. Please check.\n {ckpt_not_load}"
        print("finish loading ckpt siglip")

    def no_weight_decay(self) -> Set:
        return {"pos_embed", "cls_token", "dist_token"}

    def group_matcher(self, coarse: bool = False) -> Dict:
        return dict(
            stem=r"^cls_token|pos_embed|patch_embed",  # stem and embed
            blocks=[(r"^blocks\.(\d+)", None), (r"^norm", (99999,))],
        )

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable

    def get_classifier(self) -> nn.Cell:
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None) -> None:
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ("", "avg", "token", "map")
            if global_pool == "map" and self.attn_pool is None:
                assert False, "Cannot currently add attention pooling in reset_classifier()."
            elif global_pool != "map " and self.attn_pool is not None:
                self.attn_pool = None  # remove attention pooling
            self.global_pool = global_pool
        self.head = mint.nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x: Tensor) -> Tensor:
        if self.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
        if self.reg_token is not None:
            to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = mint.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = mint.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.pos_drop(x)

    def _intermediate_layers(
        self,
        x: Tensor,
        n: Union[int, Sequence] = 1,
    ) -> List[Tensor]:
        outputs, num_blocks = [], len(self.blocks)
        take_indices = set(range(num_blocks - n, num_blocks) if isinstance(n, int) else n)

        # forward pass
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in take_indices:
                outputs.append(x)

        return outputs

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_prefix_tokens: bool = False,
        norm: bool = False,
    ) -> Tuple[Union[Tensor, Tuple[Tensor]]]:
        """Intermediate layer accessor (NOTE: This is a WIP experiment).
        Inspired by DINO / DINOv2 interface
        """
        # take last n blocks if n is an int, if in is a sequence, select by matching indices
        outputs = self._intermediate_layers(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        prefix_tokens = [out[:, 0 : self.num_prefix_tokens] for out in outputs]
        outputs = [out[:, self.num_prefix_tokens :] for out in outputs]

        if reshape:
            grid_size = self.patch_embed.grid_size
            outputs = [
                out.reshape(x.shape[0], grid_size[0], grid_size[1], -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

        if return_prefix_tokens:
            return tuple(zip(outputs, prefix_tokens))
        return tuple(outputs)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)

        x = self.blocks(x)

        import pdb; pdb.set_trace()
        x = self.norm(x)
        return x

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.attn_pool is not None:
            x = self.attn_pool(x)
        elif self.global_pool == "avg":
            x = x[:, self.num_prefix_tokens :].mean(dim=1)
        elif self.global_pool:
            x = x[:, 0]  # class token
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def construct(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        if not self.ignore_head:
            x = self.forward_head(x)
        return x


@dataclass
class SigLIPVisionCfg:
    width: int = 1152
    layers: Union[Tuple[int, int, int, int], int] = 27
    heads: int = 16
    patch_size: int = 14
    image_size: Union[Tuple[int, int], int] = 336
    global_pool: str = "map"
    mlp_ratio: float = 3.7362
    class_token: bool = False
    num_classes: int = 0
    use_checkpoint: bool = False


SigLIP_MODEL_CONFIG = {
    "siglip_so400m_patch14_384": {
        "image_size": 336,
        "patch_size": 14,
        "width": 1152,
        "layers": 27,
        "heads": 16,
        "mlp_ratio": 3.7362,
        "global_pool": "map",
        "use_checkpoint": False,
    },
    "siglip_so400m_patch14_224": {
        "image_size": 224,
        "patch_size": 14,
        "width": 1152,
        "layers": 27,
        "heads": 16,
        "mlp_ratio": 3.7362,
        "global_pool": "map",
        "use_checkpoint": False,
    },
    "siglip_large_patch16_384": {
        "image_size": 384,
        "patch_size": 16,
        "width": 1024,
        "layers": 24,
        "heads": 16,
        "mlp_ratio": 4,
        "global_pool": "map",
        "use_checkpoint": False,
    },
    # refer to timm/models/vision_transformers.py#3633
    "vit_so400m_patch14_siglip_384": {
        "image_size": 384,
        "patch_size": 14,
        "width": 1152, # embed_dim, hidden_size
        "layers": 27, # depth
        "heads": 16,
        "mlp_ratio": 3.7362,
        "global_pool": "map",
        "use_checkpoint": False,
    }
}


def create_model(
    model_name: str = "siglip_so400m_patch14_384",
    image_size: int = None,
    layers: int = None,
    select_layer: int = -1,
    act_layer = mint.nn.GELU,
    param_dtype = ms.float32,
	keep_norm_fp32 = False,
    amp_level: str = None,
    ckpt_path: str = None,
    **kwargs,
):
    '''
    Args:
        model_name: model name
        image_size: image size, if None, use the default value in SigLIP_MODEL_CONFIG[model_name] 
        layers: number of transformer blocks, if None, use the default value in SigLIP_MODEL_CONFIG[model_name] 
		param_dtype: parameter dtype: ms.bfloat16 or ms.float32
		keep_norm_fp32: if True, LayerNorm weight and bias will be kept fp32
		amp_level: only valid if param_dtype != ms.float32, value can be None or "O2"
    '''

    assert model_name in SigLIP_MODEL_CONFIG.keys(), f"model name should be in {SigLIP_MODEL_CONFIG.keys()}"

    vision_cfg = SigLIPVisionCfg(**SigLIP_MODEL_CONFIG[model_name])

    if image_size is not None:
        vision_cfg.image_size = image_size
    if layers is not None:
        vision_cfg.layers = layers 

    if select_layer <= 0:
        layers = min(vision_cfg.layers, vision_cfg.layers + select_layer + 1)
    else:
        layers = min(vision_cfg.layers, select_layer)
    
    model = VisionTransformer(
        img_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        embed_dim=vision_cfg.width,
        depth=layers,
        num_heads=vision_cfg.heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        class_token=vision_cfg.class_token,
        global_pool=vision_cfg.global_pool,
        ignore_head=kwargs.get("ignore_head", True),
        weight_init=kwargs.get("weight_init", "skip"),
        num_classes=0,
        act_layer=act_layer,
    )

    if ckpt_path is not None:
        model.load_from_checkpoint(ckpt_path, amp_level=amp_level)
        
    # torch_dtype in transformers
    if param_dtype != ms.float32:
        set_model_param_dtype(model, dtype=param_dtype, keep_norm_fp32=keep_norm_fp32)
    
    # torch autocast
    if amp_level is not None: 
        ms.amp.custom_mixed_precision(model, dtype=param_dtype,
            black_list=[mint.nn.GroupNorm, mint.nn.LayerNorm])
    

    return model
