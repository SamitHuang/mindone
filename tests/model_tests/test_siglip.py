import os
import sys
import numpy as np

import mindspore as ms
from mindspore import Tensor, amp

from mindone.models.siglip_vit import create_model
from compare import calc_diff

np.random.seed(42)


def test(dtype=ms.bfloat16, pt_inp="siglip_inp.npy", pt_out="siglip_out.npy"):
    ckpt_path = "/home/hyx/models/timm/ViT-SO400M-14-SigLIP-384/open_clip_model.safetensors" 
    model_name = "vit_so400m_patch14_siglip_384"
    # load input and golden gt, run this testing under Janus dir
    if os.path.exists(pt_inp):
        print("Loading input from ", pt_inp)
        input_tensor = Tensor(np.load(pt_inp)).to(dtype)
    else:
        # random tensor
        shape = (1, 3, 384, 384)
        print("Create random input")
        input_tensor = np.random.normal(size=shape).astype(np.float32)
        # import pdb; pdb.set_trace()
        input_tensor = ms.Tensor(input_tensor).to(dtype)
   
    # pt_out =  "./image_forward_outs.npy"
    if os.path.exists(pt_out):
        gt_tensor = np.load()
        print(f"gt tensor dtype is {gt_tensor.dtype}")
    else:
        gt_tensor = None 

    select_layer: int = -1

    model = create_model(
        model_name,
        select_layer=select_layer,
        ckpt_path=ckpt_path,
        param_dtype=dtype,
		keep_norm_fp32=None,
		amp_level="O2",
    )

    # cal & eval
    out = model.forward_features(input_tensor)
    out = out.to(ms.float32).asnumpy()

    if gt_tensor is None:
        print('out shape: ', out.shape)
    else:
        diff = calc_diff(out, gt_tensor)
        print(diff)
        print("test finish")


if __name__ == "__main__":
    ms.set_context(mode=1)
    # test(dtype=ms.float32, pt_inp="/home/hyx/mindone/tests/model_tests/pta_vit_inp.npy")

    test(dtype=ms.bfloat16, pt_inp="/home/hyx/mindone/tests/model_tests/pta_vit_inp.npy")
