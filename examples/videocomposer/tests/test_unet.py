from vc.models import UNetSD_temporal
import mindspore as ms
from mindspore import ops
import numpy as np
from copy import deepcopy
import difflib
import time

import sys
sys.path.append("../stable_diffusion_v2/")
from ldm.util import count_params
from ldm.modules.train.tools import set_random_seed

import logging
logger = logging.getLogger(__name__)
from ldm.modules.logger import set_logger

set_logger(name="", output_dir="./tests")


def auto_map(model, param_dict):
    """Raname part of the param_dict such that names from checkpoint and model are consistent"""
    updated_param_dict = deepcopy(param_dict)
    net_param = model.get_parameters()
    ckpt_param = list(updated_param_dict.keys())
    remap = {}
    for param in net_param:
        if param.name not in ckpt_param:
            logger.info(f'Cannot find a param to load: {param.name}')
            poss = difflib.get_close_matches(param.name, ckpt_param, n=3, cutoff=0.6)
            if len(poss) > 0:
                logger.info(f'=> Find most matched param: {poss[0]},  loaded')
                updated_param_dict[param.name] = updated_param_dict.pop(poss[0])  # replace
                remap[param.name] = poss[0]
            else:
                raise ValueError('Cannot find any matching param from: ', ckpt_param)

    if remap != {}:
        logger.warning('Auto mapping succeed. Please check the found mapping names to ensure correctness')
        logger.info('\tNet Param\t<---\tCkpt Param')
        for k in remap:
            logger.info(f'\t{k}\t<---\t{remap[k]}')
    return updated_param_dict


def test():
    from vc.config.base import cfg

    set_random_seed(42)
    ms.set_context(mode=0)
    # [model] unet

    cfg.video_compositions = ["text"] #, "depthmap"]
    cfg.temporal_attention = True # try disable temporal attn and check graph and time

    model = UNetSD_temporal(
            cfg=cfg,
            in_dim=cfg.unet_in_dim,
            concat_dim=cfg.unet_concat_dim,
            dim=cfg.unet_dim,
            y_dim=cfg.unet_y_dim,
            context_dim=cfg.unet_context_dim,
            out_dim=cfg.unet_out_dim,
            dim_mult=cfg.unet_dim_mult,
            num_heads=cfg.unet_num_heads,
            head_dim=cfg.unet_head_dim,
            num_res_blocks=cfg.unet_res_blocks,
            attn_scales=cfg.unet_attn_scales,
            dropout=cfg.unet_dropout,
            temporal_attention=cfg.temporal_attention,
            temporal_attn_times=cfg.temporal_attn_times,
            use_checkpoint=cfg.use_checkpoint,
            use_fps_condition=cfg.use_fps_condition,
            use_sim_mask=cfg.use_sim_mask,
            video_compositions=cfg.video_compositions,
            misc_dropout=cfg.misc_dropout,
            p_all_zero=cfg.p_all_zero,
            p_all_keep=cfg.p_all_zero,
            #zero_y=zero_y,
            #black_image_feature=black_image_feature,
        )
    model = model.set_train(False).to_float(ms.float32)
    ckpt_path = "./model_weights/non_ema_228000.ckpt"
    #model.load_state_dict(ckpt_path, text_to_video_pretrain=False)
    param_dict = ms.load_checkpoint(ckpt_path)
    param_dict = auto_map(model, param_dict)
    param_not_load, ckpt_not_load = ms.load_param_into_net(model, param_dict, strict_load=True)
    print("Net params not load: ", param_not_load)
    print("Ckpt params not used: ", ckpt_not_load)

    #print(int(sum(p.numel() for k, p in model.named_parameters()) / (1024**2)), "M parameters")
    num_params = count_params(model)[0]
    print("UNet params: {:,}".format(num_params))
    
    # prepare inputs
    batch, c, f, h, w = 1, 4, 16, 128//8, 128//8
    latent_frames = np.ones([batch, c, f, h, w])  / 2.0
    x_t = latent_frames = ms.Tensor(latent_frames)

    txt_emb_dim = cfg.unet_context_dim
    seq_len = 77
    txt_emb = np.ones([batch, seq_len, txt_emb_dim]) / 2.0
    y = txt_emb = ms.Tensor(txt_emb)
    
    step = 50
    t = ops.full((batch,), step, dtype=ms.int64) # [t, t, ...]
    
    time_cost = []
    trials = 3
    for i in range(trials): 
        s = time.time()
        noise = model(x_t, t, y)
        dur = time.time()-s
        print("infer res: ", noise.max(), noise.min())
        print("time cost: ", dur)
        time_cost.append(dur)
    
    print("Time cost: ", time_cost)

if __name__=='__main__':
    test() 
