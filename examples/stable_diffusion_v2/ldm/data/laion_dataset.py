import gc
import logging
import os
from collections import defaultdict
from random import randint
import glob
import webdataset as wds
import mindspore as ms

import albumentations
import imagesize
import numpy as np
import pandas as pd
from ldm.data.t2i_collate import data_column, t2i_collate
from ldm.models.clip.simple_tokenizer import get_tokenizer


class LAION_Webdataset:
    """
    An IterableDataset to read laion based on Webdataset
    """
    def __init__(
        self,
        urls_or_paths, # urls or paths for tar files
        tokenizer,
        image_size,
        dataset_size=-1, # total num of data samples
        shuffle=True,
        shuffle_buffer_size=1000,
        random_crop=False,
        filter_small_size=True, small_size=512,
        filter_small_sim=True, small_sim=0.28,
        #filter_watermark=True, big_watermark=0.8,
        num_shards=1,
        shard_id=0,
        verbose=True,
    ):
        assert dataset_size>0, "Must provide the exact dataset size"
        self.dataset_size = dataset_size

        # TODO: put into a function, do sharding, a simple strategy - even distribute on tar file level (not sample level)


        # webdataset reader for large-scale data
        ds = wds.WebDataset(urls_or_paths) #, shardshuffle=shuffle) # repeat = True?
        if shuffle:
            ds = ds.shuffle(1000)
        self.ds = ds.decode("rgb8").to_tuple("jpg;png", "json") # rgb8 in range 0, 255
        self.ds.length = dataset_size # wds dataset length

        # conditions to filter unexptected samples
        self.filter_small_size = filter_small_size # some image size are not the same as annotated in metadata, e.g. shape=64x64 (<0.1%)
        self.small_size = small_size
        self.num_small_size = 0
        self.filter_small_sim = filter_small_sim # some texts are noisy and have low text-image similary (<0.1%). some texts only contain url
        self.small_sim = small_sim
        self.num_small_sim = 0
        self.num_text_url = 0

        # ~5% samples with pwatermark>0.8, but somehow the used watermark detector is not so accurate (some images with high pwatermark are actually free from waterwark).
        #self.filter_watermark = filter_watermark
        #self.big_watermark = big_watermark
        #self.num_watermark = 0

        # image and text process config
        self.tokenizer = tokenizer #if tokenizer is not None else lambda x:x
        self.image_size = image_size # target image size for training
        self.random_crop = random_crop

        #TODO: make the aug pipeline configurable
        self.rescaler = albumentations.SmallestMaxSize(max_size=self.image_size)
        if not self.random_crop:
            self.cropper = albumentations.CenterCrop(height=self.image_size, width=self.image_size)
            self.preprocessor = albumentations.Compose([self.rescaler, self.cropper])
            print("apply center crop and horizontal flip")
        else:
            self.cropper = albumentations.RandomCrop(height=self.image_size, width=self.image_size)
            self.preprocessor = albumentations.Compose(
                [self.rescaler, self.cropper, albumentations.HorizontalFlip(p=0.5)]
            )
            print("apply random crop and horizontal flip")

        # used to replace the abnormal samples
        self._replace_image = np.zeros([512, 512, 3], dtype=np.float32)
        self._replace_token = self.tokenize("")
        self.verbose = verbose

    def __len__(self):
        return self.dataset_size

    def sequential_sample(self, ind):
        return self.__iter__()

    #def __next__(self):

    def __iter__(self):
        for image, data in self.ds:
            try:
                text = data['caption']
                h, w = image.shape[:2]
                #is_text_url = text.startswith("http://") or text.startswith("https://")

                if self.filter_small_size and (min(h, w) < self.small_size):
                    self.num_small_size += 1
                    if self.verbose: print("==> Find and replace small size sample: ", text, (h, w))
                    yield (self._replace_image, self._replace_token) # yield last sample
                elif self.filter_small_sim and (data['similarity'] < self.small_sim):
                    self.num_small_sim += 1
                    if self.verbose: print("==> Find and replace low text-image similarity sample: ", text, data['similarity'])
                    yield (self._replace_image, self._replace_token) # yield last sample
                elif (text.startswith("http://") or text.startswith("https://")): # is_text_url, e.g. part_1/00000/000005967.jpg
                    self.num_text_url += 1
                    if self.verbose: print("==> Find and replace url text sample: ", text)
                    yield (self._replace_image, self._replace_token) # yield last sample
                #elif self.filter_watermark and (data['pwatermark'] is not None and data['pwatermark'] > self.big_watermark):
                #    self.num_watermark += 1
                #    if self.verbose: print("==> Find and replace high pwatermark sample: ", text, data['pwatermark'])
                #    yield (self._replace_image, self._replace_token) # yield last sample
                else:
                    # image and text preprocessing
                    image = self.preprocess_image(image)
                    caption = data['caption']
                    caption_tokens = self.tokenize(caption)

                    if self.filter_small_size or self.filter_small_sim or self.filter_watermark:
                        self._replace_image = image
                        self._replace_token = caption_tokens

                    yield (image, caption_tokens)

            except StopIteration:
                print(f"Finish itering the whole WSD, {self.ds.length} samples.\nNum small ", self.num_small)
                print("Scanned samples: ", self._index-1)
                raise StopIteration

    def preprocess_image(self, img_rgb8):
        image = self.preprocessor(image=img_rgb8)["image"]
        image = (image / 127.5 - 1.0).astype(np.float32)
        return image

    def tokenize(self, text):
        if self.tokenizer is None:
            return text

        SOT_TEXT = self.tokenizer.sot_text  # "[CLS]"
        EOT_TEXT = self.tokenizer.eot_text  # "[SEP]"
        CONTEXT_LEN = 77

        sot_token = self.tokenizer.encoder[SOT_TEXT]
        eot_token = self.tokenizer.encoder[EOT_TEXT]
        tokens = [sot_token] + self.tokenizer.encode(text) + [eot_token]
        result = np.zeros([CONTEXT_LEN])
        if len(tokens) > CONTEXT_LEN:
            tokens = tokens[: CONTEXT_LEN - 1] + [eot_token]
        result[: len(tokens)] = tokens

        return result


def _sharding(urls_or_paths, num_shards=1, shard_id=0):
    pass


def _check_and_download_tars(urls_or_paths, download_dir=''):
    def _download():
        # TODO: impl donwload from OBS or HF
        save_fp = os.path.join(download_dir, f"part_xx/xxxxxx.tar")
        raise NotImplementedError

    tar_paths = []
    for url_or_path in urls_or_paths:
        if not os.path.exists(url_or_path):
            file_path = _download(url_or_path, download_dir)
            tar_paths.append(file_path)

    return tar_paths


def read_data_stats(data_dir, remote_dataset_root=None):
    '''
    remote_dataset_root: if None, data_dir is the data root containing folders of data parts. If not None, it should be a url prefix to a remote server, e.g. for `https://huggingface.co/datasets/jasonhuang23/sd2.1_base_train/resolve/main/part_1/00000.tar`, the prefix is `https://huggingface.co/datasets/jasonhuang23/sd2.1_base_train/resolve/main`
    '''
    # data_dir/part_{id}_stats.csv
    stats_fps = glob.glob(f'{data_dir}/part_*_stats.csv') # TODO:
    assert len(stats_fps) > 0, 'No data stats csv files found. Expect part_{id}_stats.csv under data dir'
    #stat_fps = [f'{data_dir}/all_stats.csv']

    urls_or_paths = []
    sample_nums = []
    num_parts = len(stats_fps)
    for i, stats_fp in enumerate(stats_fps):
        #stats_fp = os.path.join(data_dir, f'part_{i+1}_stats.csv')
        df = pd.read_csv(stats_fp)
        tar_paths = list(df["file_path"]) # relative path of tar files
        for tar_path in tar_paths:
            if remote_dataset_root is not None:
                tar_url = remote_dataset_root + '/' + tar_path
                urls_or_paths.append(tar_url)
            else:
                tar_abs_path = os.path.join(data_dir, tar_path)
                urls_or_paths.append(tar_abs_path)
        sample_nums.extend(list(df["num_samples"]))

    return urls_or_paths, sample_nums


def load_laion_data(data_stats_dir, # local path to a directory containing data statistic files, i.e. data file relative path, number for samples.
                    batch_size,
                    tokenizer,
                    data_dir=None,
                    image_size=512,
                    shuffle=True,
                    random_crop=False,
                    filter_small_size=True,
                    small_size=512,
                    device_num=1,
                    rank_id=0,
                    num_workers=8,
                    download=False,
                    cache_dir=None,
                    ):
    '''
    A pipeline to load large-scale dataset based on IterableDataset + Sharding + Streaming for efficiency.

    Args:
        data_dir: local path or remote url path toward a directory containing training data files, e.g., /data3/laion_subset, https://my_server/laion_subset. if None, data_dir will be set the same as `data_stats_dir`.
        data_stats_dir: local path to a directory that contains csv files recording data statistic (relative file path, num samples) for all training data files, where the csv files are named in format `*_stats.csv`. Because IterableDataset requires to parse the dataset size explicitly for efficient loading, the data stats are necessary.
        download: if True, download the allocated data files from remote url (parsed from `data_dir`) before training. If False, downaload data samples needed while training (streaming). Only valid when `data_dir` is a remote url.

    Notes:
    - File structure for `data_stats_dir` should be as follows
        ```text
        data_stats_dir
        ├── part_1_stats.csv  # data statistic for part 1, i.e. image archive path, num samples in each archive
        ├── part_2_stats.csv
        ├── ...
        ├── part_64_stats.csv
        ├── part_1  # training data part 1, if `data_dir` is set as the same as `data_stats_dir`
        │   │   ├── 00000.tar # archive of images (jpg) and annotations (json)
        │   │   ├── 00001.tar
        │   │   └── ...
        ├── part_2
        ├── ....
        ```
    '''
    if data_dir is None:
        data_dir = data_stats_dir
    if cache_dir is None:
        cahce_dir = data_stats_dir

    # 1. read tar path/url list for the whole laion training set
    urls_or_paths_all, num_samples_all = read_data_stats(data_stats_dir, data_dir)
    print("Total number of tar files: ", len(urls_or_paths_all))
    print("Total dataset size: ", sum(num_samples_all))
    ##  urls_or_paths_all, num_samples_all = urls_or_paths_all[:16], num_samples_all[:16] # comment it for debug

    # 2. sharding for distributed training.
    if urls_or_paths_all is str:
        urls_or_paths_all = list(urls_or_paths_all)
    tot_tars = len(urls_or_paths_all)
    tars_per_device = tot_tars // device_num
    begin_idx = tars_per_device * rank_id
    end_idx = tars_per_device * (rank_id+1)

    urls_or_paths = urls_or_paths_all[begin_idx: end_idx]
    num_samples_list = num_samples_all[begin_idx: end_idx]
    dataset_size = sum(num_samples_list)

    print(f"Number of tar files allocated to device {rank_id}: ", len(urls_or_paths))
    print(f"Number of traininag samples for device {rank_id}: ", dataset_size)

    # 3. download the tars in `urls_or_paths` if inputs are urls and streaming is False. (optional)
    # generate list of tar file paths or urls for current shard
    if download:
        #urls_or_paths = _check_and_download_tars(urls_or_paths, download_dir)
        raise NotImplementedError("Please sync ") # TODO: impl

    # 4. build source webdataset (supporting streaming)
    # TODO: support mindrecord format
    ds = LAION_Webdataset(
        urls_or_paths,
        tokenizer,
        image_size,
        dataset_size=dataset_size,
        shuffle=shuffle,
        shuffle_buffer_size=1000, # NOTE: may adjust to be large for batch diversity
        random_crop=random_crop,
        filter_small_size=filter_small_size, small_size=small_size,
        filter_small_sim=True, small_sim=0.28,
        cache_dir=cache_dir,
        verbose=True,
    )

    # 5. build ms loader
    msds = ms.dataset.GeneratorDataset(source=ds,
                                       column_names=["img_feat", "txt_tokens"],
                                       num_shards=None, # stop sample level sharding
                                       shard_id=None,
                                       num_parallel_workers=num_workers, # TODO: optimal value
                                       )
    dataloader = msds.batch(
        batch_size,
        drop_remainder=True, # TODO: make it configurable
        num_parallel_workers=2, # TODO: optimal value
        #input_columns=["image", "text"],
        # output_columns=batch_column,
        # per_batch_map=per_batch_map, # uncommet to use inner-batch transformation
    )

    #dataloader = msdl.create_tuple_iterator(num_epochs=-1, do_copy=False) # uncommet for debugging or self-defined trainer with `value_and_grad`.

    meta_info = {'dataset_size': dataset_size, 'src_tars': urls_or_paths, 'tar_samples': num_samples_list}

    return dataloader, meta_info


