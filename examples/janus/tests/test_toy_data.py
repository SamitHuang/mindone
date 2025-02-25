import sys, os
import numpy as np
sys.path.append(".")
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
from janus.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image
import mindspore as ms
from mindspore.dataset.vision import Inter
from janus.models.modeling_vlm import MultiModalityConfig

def gen_t2i_train_sample(model_path='ckpts/Janus-Pro-1B', max_length=1088):  # 512+576
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    # prompt = "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair"
    prompt = "two dogs"
    conversation = [
        {
            "role": "<|User|>",
            "content": prompt,
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )

    vlcp = vl_chat_processor
    prompt = sft_format + vlcp.image_start_tag \
            + (vlcp.image_tag * vlcp.num_image_tokens) \
            + vlcp.image_end_tag \

    input_ids = vlcp.tokenizer.encode(prompt, add_special_tokens=True, padding="max_length", max_length=max_length, padding_side='left', truncation=True)
    input_ids = np.array(input_ids, np.int32)

    assert (input_ids == vlcp.image_id).sum() == vlcp.num_image_tokens, "text + image tokens exceeds max token length, please adjust max_length or num image token"

    attention_mask = np.ones(shape=[len(input_ids)], dtype=np.bool)
    attention_mask[input_ids==vlcp.pad_id] = 0

    image_seq_mask = np.zeros(shape=[len(input_ids)], dtype=np.bool)
    image_seq_mask[input_ids==vlcp.image_id] = 1

    # label, only train on vision seq
    ignore_index = -100  # TODO: read from config? but CE Loss didn't accept setting ignore_index
    labels = input_ids
    labels = np.where((input_ids == vlcp.image_id),
                labels,
                ignore_index,
                )
    labels = np.array(labels, np.int32)

    # data check
    config =  MultiModalityConfig.from_pretrained(model_path)
    assert input_ids.max() < config.language_config.vocab_size, "input token should be smaller than vocab size of mllm"
    assert image_seq_mask.sum() ==  vl_chat_processor.num_image_tokens

    return input_ids[None, ...], labels[None, ...], attention_masks[None, ...], image_seq_masks[None, ...], image[None, ...]

def gen_vqa_train_sample():
    pass

def gen_captioning_train_sample():
    pass


if __name__ == "__main__":
    gen_t2i_train_sample()
