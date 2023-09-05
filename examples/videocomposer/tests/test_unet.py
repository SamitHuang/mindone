from vc.models import UNetSD_temporal
import mindspore as ms
from mindspore import ops
import numpy as np

import sys
sys.path.append("../stable_diffusion_v2/")
from ldm.util import count_params


def test():
    from vc.config.base import cfg

    ms.set_context(mode=1)

    # [model] unet

    cfg.video_compositions = ["text"] #, "depthmap"]
    cfg.temporal_attention = False # try disable temporal attn and check graph and time

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

    #print(int(sum(p.numel() for k, p in model.named_parameters()) / (1024**2)), "M parameters")
    num_params = count_params(model)[0]
    print("UNet params: {:,}".format(num_params))
    
    # prepare inputs
    batch, c, f, h, w = 1, 4, 16, 128//4, 128//4
    latent_frames = np.ones([batch, c, f, h, w]) 
    x_t = latent_frames = ms.Tensor(latent_frames)

    txt_emb_dim = cfg.unet_context_dim
    seq_len = 77
    txt_emb = np.ones([batch, seq_len, txt_emb_dim]) 
    y = txt_emb = ms.Tensor(txt_emb)
    
    step = 10
    t = ops.full((batch,), step, dtype=ms.int64) # [t, t, ...]
    
    noise = model(x_t, t, y)

    print(noise.sum())

if __name__=='__main__':
    test() 
