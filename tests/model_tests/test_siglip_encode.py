import pickle
import mindspore
import mindspore as ms
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics
import numpy as np
from functools import partial
from mindspore import nn, mint
from mindone.models.siglip_vit import create_model
from mindone.models.timm import QuickGELUActivation 
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

    def __init__(self, dtype: ms.Type = ms.float32, amp_level="O2") -> None:
        super().__init__()

        ckpt_path = '/home/hyx/models/texthawk_vision/DS_Texthawk_0622_align_resume_iter10000pytorch_model.safetensors'

        # vision_config = # config.vision_config
        self.model_name = "vit_so400m_patch14_siglip_384"
        self.num_layers = 26 
        self.image_size = 448
        self.patch_size = 14
        self.selected_layers = [14, 18, 22, 26]
        self.act_layer = QuickGELUActivation
        
        self.vit = create_model(
            self.model_name,
            param_dtype=dtype,
            image_size=self.image_size,
            layers=self.num_layers,
			keep_norm_fp32=False,
			amp_level=amp_level,
            act_layer=self.act_layer,
            )
        
        # seq_length = num_patches, which is determined by image_size and patch_size
        self.position_ids = mint.arange(self.vit.seq_length).expand((1, -1))

        # TODO: the param name dosen't include "vit." prefix in mindspore. May fail to load.
        del self.vit.pos_embed
        self.vit.pos_embed = mint.nn.Embedding(self.vit.seq_length, self.vit.embed_dim, dtype=dtype)
        
        # remove unused post layernorm and head
        del self.vit.norm
        del self.vit.attn_pool
        del self.vit.fc_norm
        del self.vit.head_drop
        del self.vit.head

        self.vit.load_from_checkpoint(ckpt_path, add_prefix="vit.", amp_level=amp_level)

    def construct_v2(self, x: ms.Tensor, ):

        x = self.vit.patch_embed(x)  # mre = 0, by pta.reshape(5, 1152, 1024)
        # the following two operations are done in timm siglipvit patch_embed
        # x = x.reshape(x.shape[0], x.shape[1], -1)  # [batch, hidden_size, grid ** 2]
        # x = x.permute(0, 2, 1)  # [batch, grid ** 2, hidden_size]
        
        # due to new_key = new_key.replace('image_encoder.encoder.position_embeddings', 'encoder.vit.pos_embed')
        x = x + self.vit.pos_embed(self.position_ids)
        # contiguous() call required as 'permute' can sparsify the tensor and this breaks pipelining
        # x = x.permute(1, 0, 2).contiguous()  # [b, s, h] -> [s, b, h],
        
        
        # Mimic megatron VisionTransformerBlock forward pass
        hidden_states = x  # pre_process=True
        encoder_states = ()

        if self.selected_layers is not None: # self.take_indices in megatron 
            self.intermediates = list()
        
        # import pdb; pdb.set_trace()
        for layer_id, layer in enumerate(self.vit.blocks):

            hidden_states = layer(hidden_states)
            encoder_states = encoder_states + (hidden_states, )
            
            # TODO: just return the selected intermediate features to save time 
            if (self.selected_layers is not None) and ((layer_id + 1) in self.selected_layers):
                ln_hs = self.vit.final_layernorm(hidden_states)  # final_norm
                self.intermediates.append(ln_hs)

        hidden_states = encoder_states[-1]


        # from compare import print_diff; diff, pta_val = print_diff(encoder_states[-1].asnumpy().transpose(1,0,2), "/home/hyx/models/texthawk_vision/features/after_siglip_decoder_0_index_0.pkl")
        # import pdb; pdb.set_trace()
        
        # since selected_layers is not None, it's not run.
        # if self.selected_layers is None:  # and self.post_proces; not used in inference?
        #     hidden_states = self.final_layernorm(hidden_states)
        
        return hidden_states


    def construct(self, x, output_hidden_states=False, return_dict=True):
        # print("D--: force to overwrite mlp input") 
        # force_input = "/home/hyx/models/texthawk_vision/texthawk_ds_feature_gt_20250630/before_conv1_rank_0_index_0.pkl"
        # from compare import read_pickle_value, print_diff
        # x = ms.Tensor(read_pickle_value(force_input))
        # diff, pta_val = print_diff(x.asnumpy(), force_input)

        x = self.vit.patch_embed(x)  # mre = 0, by pta.reshape(5, 1152, 1024)
        # the following two operations are done in timm siglipvit patch_embed
        # x = x.reshape(x.shape[0], x.shape[1], -1)  # [batch, hidden_size, grid ** 2]
        # x = x.permute(0, 2, 1)  # [batch, grid ** 2, hidden_size]
        
        # due to new_key = new_key.replace('image_encoder.encoder.position_embeddings', 'encoder.vit.pos_embed')
        x = x + self.vit.pos_embed(self.position_ids)

        if output_hidden_states:
            hidden_states = ()
        else:
            hidden_states = None

        for block in self.vit.blocks:
            if output_hidden_states:
                hidden_states = hidden_states + (x,)
            x = block(x)

        if output_hidden_states:
            hidden_states = hidden_states + (x,)

        # from compare import print_diff; diff, pta_val = print_diff(x.asnumpy().transpose(1,0,2), "/home/hyx/models/texthawk_vision/features/after_siglip_decoder_0_index_0.pkl")
        # import pdb; pdb.set_trace()

        if return_dict:
            output = SimpleNamespace()
            output.last_hidden_state = x
            output.hidden_states = hidden_states
            return output

        if output_hidden_states:
            return (x, hidden_states) 

        return (x,) 

def test(dtype=ms.bfloat16, gt_inp=None, gt_out=None, profile=True):
    model = SigLIPVisionEncoder(dtype=dtype)
    
    if gt_inp is None: 
        shape = (1, 3, 448, 448)
        input_tensor = np.random.normal(size=shape).astype(np.float32)
        input_tensor = ms.Tensor(input_tensor).to(dtype)
    else:
        with open(gt_inp, "rb") as fp:
            value = pickle.load(fp)['all_images']
        input_tensor = ms.Tensor(value).to(dtype)
    
    if profile: 
        # 配置可扩展参数
        experimental_config = mindspore.profiler._ExperimentalConfig(
                            profiler_level=ProfilerLevel.Level0,
                            aic_metrics=AicoreMetrics.AiCoreNone,
                            l2_cache=False,
                            mstx=False,
                            data_simplification=False) 
        # 初始化profile
        with ms.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                                        schedule=ms.profiler.schedule(wait=1, warmup=1, active=2,
                                                repeat=1, skip_first=1),
                                        on_trace_ready=ms.profiler.tensorboard_trace_handler("./data"),
                                        profile_memory=False,
                                        experimental_config=experimental_config) as prof:
            for i in range(10):
                out = model(input_tensor)
                prof.step()
    else:
        out = model(input_tensor)

    # print(out.last_hidden_state.shape)
    # print(out.shape)

if __name__ == "__main__":
    ms.set_context(mode=1)
    test(
        gt_inp="/home/hyx/models/texthawk_vision/features/before_siglip_rank_0_index_0.pkl",
        gt_out="/home/hyx/models/texthawk_vision/features/after_siglip_decoder_0_index_0.pkl",
        profile=False,
    )
