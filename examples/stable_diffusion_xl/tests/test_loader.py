'''
To test in distributed mode:

export RANK_SIZE=2
export RANK_ID=0
python tests/test_loader.py

export RANK_ID=1
python tests/test_loader.py

'''

import time
import sys
sys.path.insert(0, "./")

from gm.data.loader import create_loader
from omegaconf import OmegaConf

import mindspore as ms

import os
os.environ["WIDS_VERBOSE"] = "1"


def test_loader(rank=0, rank_size=1):
    data_path = "datasets/custom"
    # shardlist_desc = "datasets/custom/data_info.json"

    config_file = "configs/training/sd_xl_base_finetune_910b_wds.yaml"
    config = OmegaConf.load(config_file)

    dataloader = create_loader(
        data_path=data_path,
        rank=1,
        rank_size=3,
        tokenizer=None,
        token_nums=None,
        **config.data,
    )

    ms.set_context(mode=0)

    num_batches = dataloader.get_dataset_size()
    print("num batches: ", num_batches)
    start = time.time()
    iterator = dataloader.create_dict_iterator()
    tot = 0
    # num_steps = config.data["total_step"]
    num_steps = 100
    run_steps = 0
    verbose = 1
    for i, batch in enumerate(iterator):
        if i >= num_steps:
            break
        dur = time.time() - start
        tot += dur
        run_steps += 1
        for k in batch:
            print(f"{i+1}/{num_steps}, time cost: {dur * 1000} ms")
            if verbose:
                print(batch[k]['txt'][0])
        start = time.time()

    mean = tot / run_steps
    print("Avg batch loading time: ", mean)



if __name__ == "__main__":
    rank_id = int(os.environ.get("RANK_ID", 0))
    rank_size  = int(os.environ.get("RANK_SIZE", 1))

    test_loader(rank_id, rank_size)
