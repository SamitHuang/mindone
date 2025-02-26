import sys, os
import time
import numpy as np
sys.path.append(".")
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
from janus.models import MultiModalityCausalLM, VLChatProcessor
from PIL import Image
import mindspore as ms
from mindspore.dataset.vision import Inter
from janus.train.t2i_dataset import TextImageDataset, create_dataloader_t2i


def test():
    model_path='ckpts/Janus-Pro-1B'
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)

    ds = TextImageDataset(csv_path='datasets/data_demo/jade/csvfile/image_text_en.csv',
                    data_dir='datasets/data_demo',
                    vl_chat_processor=vl_chat_processor,
                    null_prompt_prob=0.3,
                    )

    for i in range(10):
        input_ids, labels, attention_mask, image_seq_mask, image = ds.__getitem__(i)
        import pdb; pdb.set_trace()


def test_dataloader():
    model_path='ckpts/Janus-Pro-1B'
    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)

    dl = create_dataloader_t2i(
                    csv_path='datasets/data_demo/jade/csvfile/image_text_en.csv',
                    data_dir='datasets/data_demo',
                    vl_chat_processor=vl_chat_processor,
                    batch_size=2,
                    num_samples=20,
                    )

    iterator = dl.create_dict_iterator(100)
    start = 0
    for i, batch in enumerate(iterator):
        dur = time.time() - start
        for k in batch:
            print(k, batch[k].shape, batch[k].dtype)  # , batch[k].min(), batch[k].max())
        print(f"time cost: {dur * 1000} ms")
        start = time.time()


if __name__ == "__main__":
    test()
    # test_dataloader()
