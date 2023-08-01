import time
from tqdm import tqdm
from ldm.data.laion_dataset import LAION_Webdataset, load_laion_data
from ldm.models.clip.simple_tokenizer import get_tokenizer
import mindspore as ms

#urls_or_paths = '/Volumes/Extreme_SSD/sd2.1_base_train/part_1/00000.tar'
#dataset_size = 8075
urls_or_paths = '/Volumes/Extreme_SSD/sd2.1_base_train/part_1/00000_small.tar'
files_data_dir = '/data3/datasets/laion_sd2.1_base_train/'
dataset_size = 42

#tokenizer = None
image_size = 512
shuffle=True

batch_size = 4
num_workers = 1
device_num = 1
rank_id = 0

num_epoch = 1

ms.set_context(mode=0)


def test_laion_webdataset(loader_type=None):
    '''
    loader_type:
        None - iter one by one, to view raw text.
        ms - use ms GeneratorData and ds.batch to wrap, text tokenized
        custom - customed loader
    '''

    tokenizer = get_tokenizer("2.0") if loader_type is not None else None # None for view raw text

    ds = LAION_Webdataset(
        urls_or_paths,
        tokenizer,
        image_size,
        dataset_size=dataset_size,
        shuffle=shuffle,
        shuffle_buffer_size=1000,
        random_crop=False,
        filter_small_size=True, small_size=512,
        filter_small_sim=True, small_sim=0.28,
        verbose=True,
    )

    if loader_type is None:
        start = time.time()
        for ei in range(num_epoch):
            for i, (img, text) in enumerate(ds):
                print(i, img.shape, img.sum(), img.min(), img.max(), len(text), text)
        print("\nTime cost: ", time.time() - start)
    else:
        import mindspore as ms

        if loader_type=='ms':
            msds = ms.dataset.GeneratorDataset(source=ds,
                                               column_names=["img_feat", "txt_tokens"],
                                               num_shards=None,
                                               shard_id=None, # must set none for IterableDataset
                                               num_samples=16,
                                               )

            #data_iterator = ms_dataset.create_tuple_iterator(num_epochs=-1, do_copy=False)
            msdl = msds.batch(
                batch_size,
                drop_remainder=True,
                num_parallel_workers=num_workers,
                #input_columns=["image", "text"],
                # output_columns=batch_column,
                # per_batch_map=per_batch_map, # uncommet to use inner-batch transformation
            )

            data_iterator = msdl.create_tuple_iterator(num_epochs=-1, do_copy=False)
        elif loader_type=='custom':
            #import webdataset as wds
            #wdsdl = wds.WebLoader(ds, num_workers=num_workers, batch_size=batch_size)
            #data_iterator = iter(wdsdl)
            #import torch
            #data_iterator = torch.utils.data.DataLoader(ds, num_workers=num_workers, batch_size=batch_size)
            #print(next(iter(data_iterator)))
            pass

        start = time.time()
        for ei in range(num_epoch):
            for bi, batch in enumerate(data_iterator):
                images, texts = batch
                for i in range(images.shape[0]):
                    img = images[i]
                    text = texts[i]
                    print(bi*batch_size+i, img.shape, img.sum(), img.min(), img.max(), text.shape, text)
                #if i >= read_num:
                #    break
                #ret += item[0].sum()

        print("\nTime cost: ", time.time() - start)

def test_raw_dataset(data_dir):
    from ldm.data.dataset import load_data
    tokenizer = get_tokenizer("2.0")
    
    start = time.time()

    dataset = load_data(
        data_path=data_dir,
        batch_size=batch_size,
        tokenizer=tokenizer,
        image_size=image_size,
        image_filter_size=512,
        device_num=device_num,
        rank_id=rank_id,
        filter_small_size=True,
        shuffle=True,
    )
    data_iterator = dataset.create_tuple_iterator(num_epochs=-1, do_copy=False)

    t_setup = time.time() - start
    print("Num batches: ", dataset.get_dataset_size())
    print("\nSetup time: ", t_setup)
    
    for ei in range(num_epoch):
        for bi, batch in tqdm(enumerate(data_iterator)):
            images, texts = batch
            for i in range(images.shape[0]):
                img = images[i]
                text = texts[i]
                #print(bi*batch_size+i, img.shape, img.sum(), img.min(), img.max(), text.shape, text)

    print("\nTotal time cost: ", time.time() - start)

def test_build_laion_dataset(data_dir):
    #data_stats_dir = '/Volumes/Extreme_SSD/laion2b_en/sd2.1_base_train/text_image_data'
    data_stats_dir="/data3/datasets/laion_sd2.1_base_train"
    #data_dir =  data_stats_dir
    #data_dir = 'https://huggingface.co/datasets/jasonhuang23/sd2.1_base_train/resolve/main'

    tokenizer = get_tokenizer("2.0")

    start = time.time()
    dl, meta = load_laion_data(data_stats_dir=data_dir,
                               batch_size=batch_size,
                               tokenizer=tokenizer,
                               data_dir=data_dir,
                               device_num=device_num,
                               rank_id=rank_id,
                               num_workers=num_workers,
                               filter_small_size=True,
                               small_size=512,
                               shuffle=True,
                               verbose=False,
                               )
    print(meta)

    data_iterator = dl.create_tuple_iterator(num_epochs=-1, do_copy=False)

    t_setup = time.time() - start
    print("Num batches: ", dl.get_dataset_size())
    print("\nSetup time: ", t_setup)

    for ei in range(num_epoch):
        for bi, batch in tqdm(enumerate(data_iterator)):
            images, texts = batch
            for i in range(images.shape[0]):
                img = images[i]
                text = texts[i]
                #print(bi*batch_size+i, img.shape, img.sum(), img.min(), img.max(), text.shape, text)
            #if i >= read_num:
            #    break
            #ret += item[0].sum()

    print("\nTime cost: ", time.time() - start)


if __name__ == "__main__":
    #test_raw_dataset(files_data_dir)
    test_build_laion_dataset(files_data_dir)

    #test_laion_webdataset(loader_type=None)
    #test_laion_webdataset(loader_type="ms")

