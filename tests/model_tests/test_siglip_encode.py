import pickle
import mindspore as ms
import numpy as np
from functools import partial
from mindspore import nn, mint
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


class QuickGELUActivation(nn.Cell):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def construct(self, input: ms.Tensor) -> ms.Tensor:
        return input * mint.sigmoid(1.702 * input)


class SigLIPVisionEncoder(nn.Cell):

    def __init__(self, dtype: ms.Type = ms.float32) -> None:
        super().__init__()

        ckpt_path = '/home/hyx/models/texthawk_vision/DS_Texthawk_0622_align_resume_iter10000pytorch_model.safetensors'

        # vision_config = # config.vision_config
        self.model_name = "vit_so400m_patch14_siglip_384"
        self.num_layers = 26 
        self.image_size = 448
        self.patch_size = 14
        self.selected_layers = [14, 18, 22, 26]

        self.vit = create_model(self.model_name, param_dtype=dtype,
            image_size=self.image_size,
            layers=self.num_layers,
			keep_norm_fp32=False,
			amp_level="O2",
            )
        
        del self.vit.pos_embed 
        
        # seq_length = num_patches, which is determined by image_size and patch_size
        self.position_ids = mint.arange(self.vit.seq_length).expand((1, -1))  # TODO: expand? 
        # TODO: the param name dosen't include "vit." prefix in mindspore. May fail to load.
        self.vit.pos_embed = mint.nn.Embedding(self.vit.seq_length, self.vit.embed_dim, dtype=dtype)
        
        # del self.vit.mlp.act
        # self.vit.mlp.act = QuickGELUActivation

        # remove unused post layernorm and head
        del self.vit.norm
        del self.vit.attn_pool
        del self.vit.fc_norm
        del self.vit.head_drop
        del self.vit.head

        self.vit.load_from_checkpoint(ckpt_path, add_prefix="vit.")

    # TODO: support attention_mask=None
    def construct(self, x: ms.Tensor, *args, **kwargs):
        # return self.vit(x, *args, **kwargs)

        x = self.vit.patch_embed(x)  # mre = 0, by pta.reshape(5, 1152, 1024)
        # the following two operations are done in timm siglipvit patch_embed
        # x = x.reshape(x.shape[0], x.shape[1], -1)  # [batch, hidden_size, grid ** 2]
        # x = x.permute(0, 2, 1)  # [batch, grid ** 2, hidden_size]
        
        # due to new_key = new_key.replace('image_encoder.encoder.position_embeddings', 'encoder.vit.pos_embed')
        x = x + self.vit.pos_embed(self.position_ids)
        # contiguous() call required as 'permute' can sparsify the tensor and this breaks pipelining
        # x = x.permute(1, 0, 2).contiguous()  # [b, s, h] -> [s, b, h],
        
        # TODO: should parse selected_layers to the block
        # x = self.vit.blocks(x)
        
        # Mimic megatron VisionTransformerBlock forward pass
        hidden_states = x  # pre_process=True
        if self.selected_layers is not None: # self.take_indices in megatron 
            self.intermediates = list()  # TODO: this output is not used in inference, can be removed
        encoder_states = ()
        
        import pdb; pdb.set_trace()
        for layer_id, layer in enumerate(self.vit.blocks):
            # TODO: check compute steps in block in megatron
            hidden_states = layer(hidden_states)

            # since intermediates is not used, can comment out
            # if (self.selected_layers is not None) and ((layer_id + 1) in self.selected_layers):
            #     ln_hs = self.vit.final_layernorm(hidden_states)
            #     self.intermediates.append(ln_hs)

            encoder_states = encoder_states + (hidden_states, )
        import pdb; pdb.set_trace()
        hidden_states = encoder_states[-2]  # why take -2?
        
        # since selected_layers is not None, it's not run.
        # if self.selected_layers is None:  # and self.post_proces; not used in inference?
        #     hidden_states = self.final_layernorm(hidden_states)
        
        return hidden_states


def test(dtype=ms.bfloat16, gt_inp=None, gt_out=None):
    model = SigLIPVisionEncoder(dtype=dtype)
    
    if gt_inp is None: 
        shape = (1, 3, 448, 448)
        input_tensor = np.random.normal(size=shape).astype(np.float32)
        input_tensor = ms.Tensor(input_tensor).to(dtype)
    else:
        with open(gt_inp, "rb") as fp:
            value = pickle.load(fp)['all_images']
        input_tensor = ms.Tensor(value).to(dtype)
    
    out = model(input_tensor)
    print(out.last_hidden_state.shape)

if __name__ == "__main__":
    ms.set_context(mode=1)
    test(
        gt_inp="/home/hyx/models/texthawk_vision/features/before_siglip_rank_0_index_0.pkl",
        gt_out="/home/hyx/models/texthawk_vision/features/after_siglip_decoder_0_index_0.pkl",
    )
