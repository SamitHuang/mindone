# Training SD 2.x on LAION datasets

## Data Prepration

### Dependency

We will use `pyspark` to do metadata filtering and `img2dataset` to download source images. Please install the required packages by: 

For Linux,
```shell
apt-get install openjdk-8-jdk
pip install pyspark

pip install img2dataset
```

For MacOS:
```shell
brew install openjdk@11
brew install apache-spark
pip install pyspark
pip install img2dataset
```

### Data Description
We will use the following data source and filtering conditions for training data preparation.

- Link of source metadata: https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_4.5plus 
> Compared to LAION2b-en with 345GB metadata, this source has about 230GB metadata by fitlering samples with aesthetic score < 4.5.
- Filter conditions will be applied in the following processing steps.
```text
    lang=en
    aesthetic score>=4.5
    punsafe <= 0.98
    resolution >= 512x512
```

### Step 1. Download LAION 2B-en Metadata

Execute the following commands in terminal to download the whole laion 2b-en metadata.

```shell
mkdir laion_2b_en_ae4.5 && cd laion_2b_en_ae4.5
for i in {1..64}; do wget https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_4.5plus/resolve/main/2B-en-4.5_$i.parquet; done
cd ..
```

It results in 64 parquet files with 1,372,052,778 samples in total, which takes **214GB**.

An example sample is as follows:
```text
{'URL': 'https://img1.etsystatic.com/186/0/5531524/il_340x270.1310915051_5kax.jpg',
 'TEXT': 'Vintage Green Glass Swag Hanging Lamp Mid Century',
 'WIDTH': 340.0,
 'HEIGHT': 270.0,
 'similarity': 0.3491742014884949,
 'punsafe': 8.991600225272123e-06,
 'pwatermark': 0.14151343703269958,
 'AESTHETIC_SCORE': 4.751741409301758,
 'hash': 6170073934346815248,
 '__index_level_0__': 0}
```

Note that the key names can vary for different LAION subsets. For laion_art dataset, they are
```text
'URL', 'TEXT', 'WIDTH', 'HEIGHT', 'similarity', 'LANGUAGE', 'hash', 'pwatermark', 'punsafe', 'aesthetic']
```

### Step 2. Filter the Metadata  

We will exclude the samples in metadata that are not needed for training, which can save downloading time and storage space. 


```shell
python laion_filter_metadata.py --data_dir laion_2b_en_ae4.5  --output_dir laion_2b_en_filtered
```

where the filter conditions are hard-coded in the script as:
```text
WIDTH>=512
HEIGHT>=512
punsafe<=0.98
AESTHETIC_SCORE>=4.5
 ```

It results in **64 parquet files** with 340,954,340 samples in total, which takes **56GB**.

### Step 3. Download Source Images and Resize 

We will use `img2dataset` to download the image files from URLs in the filtered metadata, and resize, encode them into target format.

#### Option 1: Download by Part to Local Drives

If you have limited storage space (i.e., < 32TB), it is recommend to download the images part by part. We can divide all the images into 64 parts, each corresponding to an input parquet file (the division number also matters for multi-node distributed training) 

```shell
# download a part of the whole dataset, where part_id can be an integer in [1, 64].
sh laion_download_imgs.sh {part_id}
```
e.g. `sh laion_download_imgs.sh 1`

It takes about 20 hours to download one part with one node.

It results in
``` texts
532 subfolders, each supposed to have 10,000 images
URL download success rate: ~80% (by 20 July 2023)
Actually downloaded images in each subfolder: ~8,000 images
Total images actually donwloaded for part 1: 4.26M images (4261204)
Total size for part 1: 459GB
```

There are 64 parts in total, so they will result in ~272M images and take ~30TB by estimation. 

#### Option 2: Download at Whole to Large Local Drives 

If you have enough storage space >= 32TB, you can download all images at a time by

```shell
# download the whole dataset
sh laion_download_imgs.sh 0
```

#### Option 3: Distributed Downloading with a PySpark Cluster

If you have a cluster or have access to N machines over ssh, you can set up a PySpark cluster then download the images in a distributed way.

For detailed instructions, please refer to [distributed_img2dataset_tutorial](https://github.com/rom1504/img2dataset/blob/main/examples/distributed_img2dataset_tutorial.md) and [laion5B download images](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion5B.md#download-the-images).  

> TODO: The given two tutorials provide instructions for aws/aliyun instances, but not for aicc/modelarts. Investigate on how to depoly pyspark on modelarts or ascend servers. 


You can alos change the parameters for image resizing, saving format, etc. Please look into the `laion_download_imgs.sh` script and refer to `img2dataset` [API doc](https://github.com/rom1504/img2dataset/tree/main#api).

#### Notes:
- Some urls can become invalid. The success rate has dropped to around 80% from the day when LAION dataset was released,
- Without proxy in CN, the success rate can further drop to around 50%.
- For detailed reasons for download failures, you can check the log file in `{output_dir}/{id}_stats.json` and try to fix them (e.g. no-certifate error can be easily fixed if happened)


### Step 4. Generate Annotation File for Training 

This step is to record the image paths and their corresponding captions into csv files, used for data indexing in training.

```
python laion_to_csv.py --data_dir {path/to/image_download_folder}
```
> e.g.  `python laion_to_csv.py --data_dir /data/laion_filtered`


After the above steps, we will have the data dir as follows.
```text
data_dir
├── part_1.csv # annotation
├── part_1
│   ├── 00000.csv 
│   ├── 00000 
│   │   ├── 000000000.jpg
│   │   ├── 000000001.jpg 
│   │   ├── 000000002.jpg
│   │   └── ... 
│   ├── 00001.csv 
│   ├── 00001 
│   │   ├── 000010000.jpg
│   │   ├── 000010002.jpg 
│   │   └── ... 
│   ├── ... 
│       
├── part_2.csv
├── part_2
│   ├── 00000.csv 
│   ├── 00000 
│   │   ├── 000000000.jpg
...
```

## Training

### Training on Ascend Servers

After the data preparation, set up the `data_path` in `scripts/run_train_v2_distributed.sh`

Generate the hccl rank table file referring to [this doc](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/tutorials/distribute_train.md#12-configure-rank_table_file-for-training), then modify parallel config including device num in `scripts/run_train_v2_laion.sh` accordingly. 

To launch distributed training, run

```
sh scripts/run_train_v2_laion.sh
```

Note: Large global batch size is preferred for better model convergence. **2048** is a reference for producing good training results. An example setting to reach it: 64 NPUs with batch_size=3 and accumulation_steps=10.


### Training on AICC or ModelArts 

#### 1. Upload the data dir to OBS (skip it if option 3, distributed download on AICC, in Data Preparation Step 3 is achieved.)

#### 2. Set up training job on the webpage

Under development... Please refer to the documents from AICC providers.  



------------------------------- 

## Part A. Training Stable Diffusion on LAION-art dataset

LAION-art is a 8M laion5B subset with aesthetic > 8, pwatermark < 0.8, punsafe < 0.5. 

We will use it as an example to illustrate how to train a SD model on LAION dataset.

### 1. Download metadata in parquet format

Download the metadata from https://huggingface.co/datasets/laion/laion-art/ manually or via the following script. 

```shell
mkdir laion-art && cd laion-art
wget https://huggingface.co/datasets/laion/laion-art/resolve/main/laion-art.parquet
```

The parquet files contain metadata including image shape, text, language, etc. Here is a data sample read from the parquet file.

```text
{'URL': 'https://www.advocate-art.com/system/ART/Modules/Application/Images/Image/images/000/011/121/artistique_half/VML175107.jpg?6f71107bead7921d03fa7dc3e4ac4b9a8f24dfd3d823d512d7618c06e4059513',
 'TEXT': 'Christmas Shopping Copy',
 'WIDTH': 850,
 'HEIGHT': 850,
 'similarity': 0.2666246294975281,
 'LANGUAGE': 'nolang',
 'hash': -3604776403351267688,
 'pwatermark': 0.0396263524889946,
 'punsafe': 0.00027811527252197266,
 'aesthetic': 8.352225303649902}
```

### 2. Filtering the Metadata

You may filter unnecessary data to get a smaller subset training, which is useful for saving disk space. 

```
python laion_filter_metadata.py
```

The default filtering conditions in this script are: `LANGUAGE==en`, `WIDTH>=512`, and `HEIGHT>=512`. The resulting metadata should contain **947,191** samples after filtering.

**Notes:**

1. Since OpenCLIP or CLIP is only trained on English text-image data, we need to filter out non-english data via the `LANGUAGE` field to train the Stable Diffusion models for english text-to-image generation.

2. For finetuning based on SD 2.0-base, we prefer images with resolution >= 512x512.

3. To change the filtering conditions, e.g. higher aesthetic, please modified the line of code `df = df.filter(...)` in `laion_filter_metadata.py` accordingly.

### 3. Download and Resize Images 

We will use `img2dataset` to download the image files from URL, resize images, and encode them to local storage.

```shell
output_format="files"
input_folder=/data3/datasets/laion_art_metadata_filtered 
output_folder=/data3/datasets/laion_art_filtered # make sure this folder is set on the disk with large enough space
timeout=10
#encode_quality=95

img2dataset --url_list $input_folder --input_format "parquet" \
        --url_col "URL" --caption_col "TEXT" \
		--output_format $output_format \
        --output_folder  $output_folder \
		--processes_count 16 --thread_count 64 --image_size 512 \
        --resize_only_if_bigger=True \
		--resize_mode="keep_ratio" \
		--skip_reencode=True \
        --timeout $timeout \
        --save_additional_columns '["similarity","hash","punsafe","pwatermark","aesthetic","LANGUAGE"]' \
		#--enable_wandb True

```

It will take about 1 hour to finish downloading (depending on network speed). And the downloaded files should be stored in the following format:

```text
00000
├── 000000000.jpg
├── 000000001.jpg
├── 000000002.jpg
├── ... 
00001
├── 000010000.jpg
├── 000010001.jpg
├── 000010002.jpg
├── ... 
```

For more usages of img2dataset, please read the official [API doc](https://github.com/rom1504/img2dataset/tree/main#api).

**Notes**

1. You may change `output_format` to fit your tradeoff between storage space and data loading speed. The options are:
    - files:  saves as a set of subfolder containing pictures (in jpg format by default).
    - webdataset: saves as tars containing pictures, which is compressed and is fast in dataloading.
    - parquet: saves as parquet containing pictures as bytes.
2. For "failed to download" message, please checkout `{output_dir}/0000x_stats.json` for detailed reasons. Here are some solutions to increase the download success rate.
    - To address "certificate verify failed", please replace `/your/path/to/python3.x/site-packages/img2dataset/downloader.py` by `tools/downloaders/downloader.py` to set no certificate context. 
    - Use [DNS resolver](https://github.com/rom1504/img2dataset/tree/main#setting-up-a-high-performance-dns-resolver)
    - TODO: address more download failures in CN network environment.

3. For failed to resize message, you may set `--resize_mode` as "no" to disable resizing, or ajust the reszing parameters.


### 4. Precompute text embeddings and latent images (Optional)

It can reduce the storage size to about 1/4 and save training time, despite of being less flexible.

TODO: 
[] save text embedding from clip text encoder output, image embedding from AE encoder output, as mindrecord
[] dataloader to load embedding data
[] trainer supporting embeding data input


### 5. Convert to trainable data format 

This step is to gather the image path and caption pairs to form the training data.

```
python laion_to_csv.py
```


### 6. Distributed Training

Generate the hccl rank table file referring to [this doc](https://github.com/mindspore-lab/mindocr/blob/main/docs/en/tutorials/distribute_train.md#12-configure-rank_table_file-for-training).

Then modify the paths and device num in `scripts/run_train_v2_laion.sh` according to your local env. Run: 

```
sh scripts/run_train_v2_laion.sh
```

It is recommended to run training on a large number of devices (e.g. 128 NPUs), in order to reach a large `batch_size` for GD optimization. A global batch size of **2048** is a reference for producing good training results. 


## Part B. Reproducing SD 2.1-base by finetuning SD 2.0-base on LAION 5B subsets 

The overall pipeline for training on a larger LAION subsets is almost the same as Part A, except for downloading large number of metadata files, image files, and store them efficiently. 

### 2. Filter the Metadata

```
python laion_filter_metadata.py
```

Before running, please modify the following setting in the script according to your environment: 
``` text
    data_path_or_dir='/data3/datasets/laion_2b_en_ae4.5'
    num_repartitions = 50
    output_dir = '/data3/datasets/laion_2b_en_ae4.5_metadata_filtered'
```

### 3. Download and Resize Images 

We will use `img2dataset` to download the image files from URL, resize images, and encode them to local storage.

```shell
output_format="files"
input_folder=/data3/datasets/laion_2b_en_ae4.5_metadata_filtered 
output_folder=/data3/datasets/laion_2b_en_ae4.5_filtered # make sure this folder is set on the disk with large enough space
timeout=10
#encode_quality=95

img2dataset --url_list $input_folder --input_format "parquet" \
        --url_col "URL" --caption_col "TEXT" \
		--output_format $output_format \
        --output_folder  $output_folder \
		--processes_count 16 --thread_count 64 --image_size 512 \
        --resize_only_if_bigger=True \
		--resize_mode="keep_ratio" \
		--skip_reencode=True \
        --timeout $timeout \
        --save_additional_columns '["similarity","hash","punsafe","pwatermark","aesthetic","LANGUAGE"]' \
		#--enable_wandb True
```

It will take about 1 hour to finish downloading (depending on network speed). And the downloaded files should be stored in the following format:


TBC
