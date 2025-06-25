import os
import sys
import numpy as np

import mindspore as ms
from mindspore import Tensor, amp

from mindone.models.siglip_vit import create_model
from utils import diff_res

np.random.seed(42)


def test(dtype=ms.bfloat16, pt_inp="siglip_inp.npy", pt_out="siglip_out.npy"):
    ckpt_path = "/home/hyx/models/timm/ViT-SO400M-14-SigLIP-384/open_clip_model.safetensors" 
    model_name = "vit_so400m_patch14_siglip_384"

    # load input and golden gt, run this testing under Janus dir
    if os.path.exists(pt_inp):
        input_tensor = Tensor(np.load()).to(dtype)
    else:
        # random tensor
        shape = (1, 3, 384, 384)
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
        param_dtype=dtype,
        ckpt_path=ckpt_path,
    )

    # cal & eval
    out = model(input_tensor)
    out = out.to(ms.float32).asnumpy()

    if gt_tensor is None:
        print('out shape: ', out.shape)
    else:
        diff = diff_res(out, gt_tensor)
        print(diff)
        print("test finish")


if __name__ == "__main__":
    ms.set_context(mode=1)
    test()
