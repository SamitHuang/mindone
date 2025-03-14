import bisect

import mindspore as ms
import numpy as np
from mindspore.dataset import WeightedRandomSampler

from janus.models import VLChatProcessor
from janus.train.t2i_dataset import TextImageDataset
from janus.train.text_dataset import TextDataset
from janus.train.vqa_dataset import VqaDataset


class  MixDataset:
    """
    Mixed dataset that outputs pure text, multi-modal and text-to-image data.
    """
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def __init__(self, datasets, default_image_shape=(1, 3, 384, 384), max_token_length=1024):
        self.default_image_shape = default_image_shape
        self.max_token_length = max_token_length
        self.datasets = datasets
        self.num_dataset = len(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx ==0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        ret = self.datasets[dataset_idx][sample_idx]

        # add image and image_seq_mask item to pure text for batching
        if dataset_idx == 0:
            image = np.zeros(self.default_image_shape, np.float32)
            image_seq_mask = np.zeros((self.max_token_length), dtype=np.bool)
            ret += (image, image_seq_mask)

        return ret

    def __len__(self):
        return self.cumulative_sizes[-1]


def create_mix_dataloader(
                          vl_chat_processor,
                          t2i_csv_path="datasets/jade/csvfile/image_text.csv",
                          t2i_data_dir="./",
                          text_dataset_name="pubmedqa",
                          text_data_dir="datasets/PubMedQA",
                          vqa_dataset_name="medical-vqa",
                          vqa_data_dir="rbojia/medical-vqa",
                          max_token_length=1024,
                          image_size=384,
                          null_prompt_prob=0.0,
                          batch_size=1,
                          num_parallel_workers=1,
                          rank=0,
                          rank_size=1,
                          num_samples=100,
                          sample_ratios=(5, 4, 1)):

    dataset_text = TextDataset(
        dataset_name="pubmedqa",
        data_dir="datasets/PubMedQA",
        vl_chat_processor=vl_chat_processor,
        max_token_length=max_token_length,
        num_samples=num_samples,
    )

    dataset_t2i = TextImageDataset(
        csv_path=t2i_csv_path,
        data_dir=t2i_data_dir,
        vl_chat_processor=vl_chat_processor,
        max_token_length=max_token_length,
        image_size=image_size,
        null_prompt_prob=null_prompt_prob,
        num_samples=num_samples,
    )

    dataset_vqa = VqaDataset(
        dataset_name=vqa_dataset_name,
        data_dir=vqa_data_dir,
        vl_chat_processor=vl_chat_processor,
        max_token_length=max_token_length,
        num_samples=num_samples,
    )

    datasets =  [dataset_text, dataset_t2i, dataset_vqa]
    mix_dataset = MixDataset(datasets=datasets,
                             default_image_shape=(1, 3, image_size, image_size),
                             max_token_length=max_token_length)

    sample_weights = []
    assert len(sample_ratios) == len(datasets)
    for i in range(len(sample_ratios)):
        weight = sample_ratios[i] * len(mix_dataset) / len(datasets[i])
        sample_weights += [weight] * len(datasets[i])

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    dataloader = ms.dataset.GeneratorDataset(
        source=mix_dataset,
        sampler=sampler,
        column_names=["task_type", "input_ids", "labels", "attention_mask", "image_seq_mask", "image"],
        shuffle=False,
        num_parallel_workers=num_parallel_workers,
        python_multiprocessing=True,
        num_shards=rank_size,
        shard_id=rank,
    )

    dataloader = dataloader.batch(batch_size, drop_remainder=True)

    return dataloader

if __name__ == "__main__":
    pretrain_model_path = "/mnt/disk2/fredhong/hf_ckpts/Janus-Pro-1B"
    vl_chat_processor = VLChatProcessor.from_pretrained(pretrain_model_path)
    dataloader = create_mix_dataloader(vl_chat_processor)
    for data in dataloader.create_dict_iterator():
        print(data)
        break
