import sys
mindnlp_lib_path ="/home/hyx/mindnlp"
sys.path.insert(0, mindnlp_lib_path)

sys.path.append(".")

import mindspore as ms
import numpy as np
from PIL import Image
from common import calc_diff

from src.internlm_xcomposer2d5_7b.build_mlp import CLIPVisionTower

def test(mode=0, jit_level=None, pt_npy=None):
    ms.set_context(mode=mode)
    if jit_level is not None:
        ms.set_context(jit_config={"jit_level": jit_level})

    images = [ms.Tensor(np.load('tests/images_0.npy'))]
    plora_glb_GN = np.load('tests/plora_glb_GN.npy')
    plora_sub_GN = np.load('tests/plora_sub_GN.npy')
    
    plora_glb_GN = ms.Tensor(plora_glb_GN)
    plora_sub_GN = ms.Tensor(plora_sub_GN)


    vit = CLIPVisionTower('models/internlm/internlm-xcomposer2d5-clip')
    vit.set_train(False)

    output_imgs, output_len  = vit(images, plora_glb_GN, plora_sub_GN) 
    print(output_imgs) 
    print(output_len) 
    ms_res = output_imgs.asnumpy()
    
    if pt_npy is not None:
        pt_res = np.load(pt_npy)
        print(calc_diff(ms_res, pt_res))
    
if __name__ == "__main__":
    test(1, 'O0', 'tests/pt_vision_tower_out.npy')
