# TODO: use trained checkpoint and real data to check.

import sys
import torch
sys.path.append(".")
import mindspore as ms
import numpy as np
from ae.models.causal_vae_3d import Encoder, Decoder

args = dict(
    ch=128,
    out_ch=3,
    ch_mult=(1, 2, 4, 4),
    num_res_blocks=2,
    attn_resolutions=[16],
    dropout=0.0,
    resamp_with_conv=True,
    in_channels=3,
    resolution=256,
    z_channels=16,
    double_z=True,
    use_linear_attn=False,
    attn_type="vanilla3D", # diff 3d
    dtype=ms.float32,
    time_compress=2,  # diff 3d
    upcast_sigmoid=False,
    )

bs, cin, T, H, W = 1, 3, 8, 256, 256
x = np.random.normal(size=(bs, cin, T, H, W ))

def test_net_ms(x, ckpt=None, net_class=Encoder):
    net_ms = net_class(**args)
    net_ms.set_train(False)
    if ckpt:
       sd = ms.load_checkpoint(ckpt)
       m, u = ms.load_param_into_net(net_ms, sd)
       print("net param not loaded: ", m)
       print("ckpt param not loaded: ", u)

    res_ms = net_ms(ms.Tensor(x, dtype=ms.float32))
    total_params = sum([param.size for param in net_ms.get_parameters()])
    print("ms total params: ", total_params)

    print(res_ms.shape)
    return res_ms.asnumpy(), net_ms 

def test_net_pt(x, ckpt=None, save_ckpt_fn=None, net_class=None):
    net_pt = net_class(**args)
    net_pt.eval()
    if ckpt is not None:
        checkpoint = torch.load(ckpt)
        net_pt.load_state_dict(checkpoint['model_state_dict'])

    if save_ckpt_fn:
        torch.save({'model_state_dict': net_pt.state_dict(),
                    }, f"tests/{save_ckpt_fn}.pth")

    res_pt = net_pt(torch.Tensor(x))

    total_params = sum(p.numel() for p in net_pt.parameters())
    print("pt total params: ", total_params)
    print(res_pt.shape)

    return res_pt.detach().numpy(), net_pt

def _convert_ckpt(pt_ckpt):
    # sd = torch.load(pt_ckpt, map_location="CPU")['model_state_dict'] 
    sd = torch.load(pt_ckpt)['model_state_dict'] 
    target_data = []

    # import pdb
    # pdb.set_trace()

    for k in sd:
        if '.' not in k:
            # only for GroupNorm
            ms_name = k.replace("weight", "gamma").replace("bias", "beta")
        else:
            if 'norm' in k:
                ms_name = k.replace(".weight", ".gamma").replace(".bias", ".beta")
            else:
                ms_name = k
        target_data.append({"name": ms_name, "data": ms.Tensor(sd[k].detach().numpy())})

    save_fn = pt_ckpt.replace(".pth", ".ckpt")
    ms.save_checkpoint(target_data, save_fn)

    return save_fn

def _diff_res(ms_val, pt_val):
    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()
    return mae, max_ae

def compare_encoder():
    pt_code_path = "/home/mindocr/yx/Open-Sora-Plan/" 
    sys.path.append(pt_code_path)
    from opensora.models.ae.videobase.causal_vae.modeling_causalvae import Encoder as Encoder_PT 
    ckpt_fn = 'encoder'
    pt_res, net_pt = test_net_pt(x, save_ckpt_fn=ckpt_fn, net_class=Encoder_PT)
    print("pt out range: ", pt_res.min(), pt_res.max())

    ckpt = _convert_ckpt(f"tests/{ckpt_fn}.pth")    

    ms_res, net_ms = test_net_ms(x, ckpt=ckpt, net_class=Encoder)
    print(_diff_res(ms_res, pt_res))
    # (0.0001554184, 0.0014244393)

def compare_decoder():
    z_shape = (1, 16, 2, 32, 32)  # b c t h w
    z = np.random.normal(size=z_shape)

    pt_code_path = "/home/mindocr/yx/Open-Sora-Plan/" 
    sys.path.append(pt_code_path)
    from opensora.models.ae.videobase.causal_vae.modeling_causalvae import Decoder as Decoder_PT 
    ckpt_fn = 'decoder'
    pt_res, net_pt = test_net_pt(z, save_ckpt_fn=ckpt_fn, net_class=Decoder_PT)
    print("pt out range: ", pt_res.min(), pt_res.max())

    ckpt = _convert_ckpt(f"tests/{ckpt_fn}.pth")    

    ms_res, net_ms = test_net_ms(z, ckpt=ckpt, net_class=Decoder)
    print(_diff_res(ms_res, pt_res))
    # (0.0001554184, 0.0014244393)



if __name__ == "__main__":
    # test_encoder(x)
    # compare_encoder()
    compare_decoder()
