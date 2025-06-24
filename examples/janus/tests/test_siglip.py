import os
import sys
import numpy as np

import mindspore as ms
from mindspore import Tensor, amp

sys.path.append(".")
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../..")))  # for mindone

# default setup, unit load hard to load ckpt this way, do entire model loading
from janus.models.siglip_vit import create_siglip_vit
from utils import diff_res

np.random.seed(42)


def test(dtype=ms.bfloat16):
    # ckpt_path = "/home/hyx/models/Janus-Pro-1B/pytorch_model.bin" 
    # model_name: str = "siglip_large_patch16_384"

    ckpt_path = "/home/hyx/models/timm/ViT-SO400M-14-SigLIP-384/open_clip_model.safetensors" 
    model_name = "vit_so400m_patch14_siglip_384"

    # load input and golden gt, run this testing under Janus dir
    img_tensor_path = "./image_tensor.npy"
    if os.path.exists(img_tensor_path):
        input_tensor = Tensor(np.load()).to(dtype)
    else:
        # random tensor
        shape = (1, 3, 384, 384)
        input_tensor = np.random.normal(size=shape).astype(np.float32)
        # import pdb; pdb.set_trace()
        input_tensor = ms.Tensor(input_tensor).to(dtype)
   
    gt_path =  "./image_forward_outs.npy"
    if os.path.exists(gt_path):
        gt_tensor = np.load()
        print(f"gt tensor dtype is {gt_tensor.dtype}")
    else:
        gt_tensor = None 


    select_layer: int = -1
    vision_tower = create_siglip_vit(
        model_name,
        select_layer=select_layer,
    )
    vision_tower.load_from_checkpoint(ckpt_path)

    print(f"dtype conversion is using with {dtype}")

    # if dtype != ms.float32:
    #     set_model_param_dtype(vision_tower, dtype=dtype, keep_norm_fp32=False)
    if dtype != ms.float32:
        amp.auto_mixed_precision(vision_tower, amp_level="O2", dtype=dtype)

    # cal & eval
    out = vision_tower(input_tensor)
    out = out.to(ms.float32).asnumpy()

    # assert np.allclose(out, gt_tensor), f"recal result is not closed to gt!, out:{out.shape}\n{out}\ngt:{gt_tensor.shape}\n{gt_tensor}"

    if gt_tensor is None:
        print('out shape: ', out.shape)
    else:
        diff = diff_res(out, gt_tensor)
        print(diff)
        print("test finish")


if __name__ == "__main__":
    # ms.set_context(device_id=7, mode=1, pynative_synchronize=True)
    ms.set_context(mode=1)
    test()
