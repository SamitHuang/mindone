import sys
mindnlp_lib_path ="/home/hyx/mindnlp"
sys.path.insert(0, mindnlp_lib_path)

import mindspore as ms
import numpy as np
from mindnlp.transformers import AutoModel, CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from PIL import Image
from common import calc_diff


def test(mode=0, jit_level=None, pt_npy=None):
    ms.set_context(mode=mode)
    if jit_level is not None:
        ms.set_context(jit_config={"jit_level": jit_level})

    # import ipdb; ipdb.set_trace()

    image_proc = CLIPImageProcessor.from_pretrained("models/internlm/internlm-xcomposer2d5-clip")
    img = Image.open('examples/cars1.jpg')
    pixel_values = image_proc(images=img, return_tensors='ms').pixel_values

    # model = CLIPVisionModel.from_pretrained("models/internlm/internlm-xcomposer2d5-clip")
    model = AutoModel.from_pretrained("models/internlm/internlm-xcomposer2d5-clip")

    model.set_train(False)
    
    outputs = model(pixel_values=pixel_values, output_hidden_states=True)

    print(outputs.hidden_states[-1].shape)
    ms_res = (outputs.hidden_states[-1]).asnumpy()
    np.save('ms_clip_out.npy', ms_res)
    
    if pt_npy is not None:
        pt_res = np.load(pt_npy)
        print(calc_diff(ms_res, pt_res))


if __name__ == "__main__":
    test(0, 'O0', pt_npy = 'tests/pt_clip.npy')

