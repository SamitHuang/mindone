# Training SD 2.x on LAION datasets

## Data Preparation

Please refer to [LAION Subst Preparation](tools/data_utils/README.md)


Once ready, you should have the data  
```text
data_dir
├── part_1_stats.csv # annotation, record: tar file path, number of samples
├── part_1/
│   ├── 00000.tar ( archive of image files and the corresponding text and metadata as follows)
│   │   ├── 000000000.jpg
│   │   ├── 000000000.json
│   │   ├── 000000000.text
│   │   ├── 000000001.jpg
│   │   ├── 000000001.json
│   │   ├── 000000001.text
│   │   └── ...
│   ├── ...
│     
├── part_2_stats.csv
├── part_2/
...
```

## Training

## Train on 

### Training on Ascend Servers

After the data preparation, set up the `data_path` in `scripts/run_train_v2_distributed.sh`

Generate the hccl rank table file referring to [this doc](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/tutorials/distribute_train.md#12-configure-rank_table_file-for-training), then modify parallel config including device num in `scripts/run_train_v2_laion.sh` accordingly. 

To launch distributed training, run

```
sh scripts/run_train_v2_laion.sh
```

Note: Large global batch size is preferred for better model convergence. **2048** is a reference for producing good training results. An example setting to reach it: 64 NPUs with batch_size=3 and accumulation_steps=10.


### Training on AICC or ModelArts (coming soon)

#### 1. Upload the data dir to OBS (skip it if option 3, distributed download on AICC, in Data Preparation Step 3 is achieved.)

#### 2. Set up training job on the webpage

Under development. Please seek documents from AICC providers.  

