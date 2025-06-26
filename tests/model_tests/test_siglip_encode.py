import mindspore as ms
import numpy as np
from functools import partial
from mindspore import nn
from mindone.models.siglip_vit import create_model
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Set, Tuple


"""
model config in training:
num_layers: 26
hidden_size: 1152
num_attention_head: 16
num_query_group: 16
ffn_hidden_size: 4304
"""

class SigLIPVisionEncoder(nn.Cell):

    def __init__(self, dtype: ms.Type = ms.float32) -> None:
        super().__init__()

        # vision_config = # config.vision_config
        self.model_name = "vit_so400m_patch14_siglip_384"
        self.num_layers = 26 
        self.image_size = 448
        self.patch_size = 14
        self.select_layers = [14, 18, 22, 26]

        self.vit = create_model(self.model_name, param_dtype=dtype,
            image_size=self.image_size,
            layers=self.num_layers,
			keep_norm_fp32=True,
			amp_level="O2",
            )

        # substitude patch embedding
        '''
        In training code, pos_embed is defined as:
        self.num_patches_per_dim_h = self.img_h // self.patch_size
        self.num_patches_per_dim_w = self.img_w // self.patch_size
        self.num_patches = self.num_patches_per_dim_h * self.num_patches_per_dim_w
        self.seq_length = self.num_patches
        self.position_embeddings = torch.nn.Embedding(self.seq_length, self.visual_hidden_size)

        which means seq_length is determined by patch_size and image_size

        PatchEmbed is defined as:
			self.conv1 = torch.nn.Conv2d(
				in_channels=3,
				out_channels=self.visual_hidden_size,
				kernel_size=self.patch_size,
				stride=self.patch_size,
				bias=True,
				padding='valid'
			)
        '''

        # remove unused layers
        print(slice(self.num_layers, len(self.vit.blocks)))

        # remove unused post layernorm and head
        del self.vit.norm
        del self.vit.attn_pool
        del self.vit.fc_norm
        del self.vit.head_drop
        del self.vit.head

        # modify forward function to get intermediate hidden states
        def custom_construct(
            self,
            x: ms.Tensor,
            output_hidden_states: bool = False,
            return_dict: bool = True,
        ):
            x = self.patch_embed(x)
            x = self._pos_embed(x)
            x = self.patch_drop(x)
            x = self.norm_pre(x)
            if output_hidden_states:
                hidden_states = ()
            else:
                hidden_states = None

            for blk in self.blocks:
                if output_hidden_states:
                    hidden_states = hidden_states + (x,)
                x = blk(x)
            if output_hidden_states:
                hidden_states = hidden_states + (x,)

            if return_dict:
                output = SimpleNamespace()
                output.last_hidden_state = x
                output.hidden_states = hidden_states
                return output

            if output_hidden_states:
                return (x, hidden_states)
            return (x,)

        self.vit.construct = partial(custom_construct, self.vit)

    def construct(self, x: ms.Tensor, *args, **kwargs):
        return self.vit(x, *args, **kwargs)

def test(dtype=ms.bfloat16):
    model = SigLIPVisionEncoder(dtype=dtype)
    
    shape = (1, 3, 448, 448)
    input_tensor = np.random.normal(size=shape).astype(np.float32)
    input_tensor = ms.Tensor(input_tensor).to(dtype)
    
    out = model(input_tensor)
    print(out.last_hidden_state.shape)

if __name__ == "__main__":
    ms.set_context(mode=1)
    test()
