# flake8: noqa
import math
import os
import sys
import time

import numpy as np
from easydict import EasyDict as edict
from PIL import Image

import mindspore as ms
from mindspore import amp, ops
from mindspore.nn.utils import no_init_parameters
from mindspore.ops.operations.nn_ops import FlashAttentionScore

sys.path.insert(0, ".")
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)

# from mindone.transformers import AutoProcessor
from transformers import AutoProcessor

from mindone.transformers.models.llava import LlavaForConditionalGeneration
from mindone.transformers.models.llava.configuration_llava import LlavaConfig

from hyvideo.utils.helpers import set_model_param_dtype


def test():
    model_path = "ckpts/llava-llama-3-8b-v1_1-transformers"
    dtype = ms.float16

    # model_path = 'ckpts/llava_tiny'
    processor = AutoProcessor.from_pretrained(model_path)

    prompt = (
        "<|start_header_id|>user<|end_header_id|>\n\n<image>\nWhat are these?<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    image_file = "./000000039769.jpg"
    raw_image = Image.open(image_file)

    inputs = processor(text=prompt, images=raw_image, return_tensors="np")
    inputs["input_ids"] = ms.tensor(inputs["input_ids"], dtype=ms.int32)
    inputs["attention_mask"] = ms.tensor(inputs["attention_mask"], dtype=ms.bool_)
    inputs["pixel_values"] = ms.tensor(inputs["pixel_values"]).to(dtype)


    feature_only = False
    config = LlavaConfig.from_pretrained(model_path)
    config.text_config._attn_implementation = "flash_attention_2"
    model = LlavaForConditionalGeneration.from_pretrained(model_path, text_config=config.text_config)

    if dtype != ms.float32:
        set_model_param_dtype(model, dtype=dtype)

    if feature_only:
        outputs = model(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            attention_mask=inputs["attention_mask"],
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        print("num hidden state: ", len(outputs.hidden_states[-1]))
        print("last hidden state: ", outputs.hidden_states[-1].shape)
        # import pdb; pdb.set_trace()
    else:
        output = model.generate(**inputs, max_new_tokens=200 + 824, do_sample=False, use_cache=False)

        print(processor.decode(output[0][2:], skip_special_tokens=True))


if __name__ == "__main__":
    ms.set_context(mode=1)
    test()
