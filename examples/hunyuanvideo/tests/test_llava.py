# flake8: noqa
import math
import os
import sys
import time

from PIL import Image
import numpy as np
from easydict import EasyDict as edict

import mindspore as ms
from mindspore.nn.utils import no_init_parameters
from mindspore import amp, ops
from mindspore.ops.operations.nn_ops import FlashAttentionScore

sys.path.insert(0, ".")
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)

from mindone.transformers.models.llava.configuration_llava import LlavaConfig
from mindone.transformers.models.llava import LlavaForConditionalGeneration

# from mindone.transformers import AutoProcessor
from transformers import AutoProcessor

def test():
    # model_path = 'ckpts/llava-llama-3-8b-v1_1-transformers'
    model_path = 'ckpts/llava_tiny'
    processor = AutoProcessor.from_pretrained(model_path)
    
    # run
    prompt = ("<|start_header_id|>user<|end_header_id|>\n\n<image>\nWhat are these?<|eot_id|>"
              "<|start_header_id|>assistant<|end_header_id|>\n\n")
    image_file = "./000000039769.jpg"
    raw_image = Image.open(image_file)

    inputs = processor(text=prompt, images=raw_image, return_tensors='np') # .to(ms.float16)
    inputs['input_ids'] = ms.tensor(inputs['input_ids'], dtype=ms.int32)
    inputs['attention_mask'] = ms.tensor(inputs['attention_mask'], dtype=ms.bool_)
    inputs['pixel_values'] = ms.tensor(inputs['pixel_values']).to(ms.float16)
    
    # inputs .to(ms.float16)

    # TODO: check mixed precision setting, to(float16) may not be what we want
    debug = True
    if debug:
        with no_init_parameters():
            config = LlavaConfig.from_pretrained(model_path, mindspore_dtype=ms.float16)
            model = LlavaForConditionalGeneration(config=config)
        outputs = model(
            input_ids=inputs['input_ids'], pixel_values=inputs['pixel_values'], attention_mask=inputs['attention_mask'],
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            )
        import pdb; pdb.set_trace()
        print(outputs)
    else:
        model = LlavaForConditionalGeneration.from_pretrained(config=config, mindspore_dtype=float16)

        output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        print(processor.decode(output[0][2:], skip_special_tokens=True))


if __name__ == "__main__":
    ms.set_context(mode=1)
    test()
        
