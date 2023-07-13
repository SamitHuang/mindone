# Training SD 2.x on LAION datasets

#### Dependency

It is recommended to use `pyspark` to do metadata filtering and use `img2dataset` to download source images in jparallel. Please install dependency by: 

```shell
apt-get install openjdk-8-jdk
pip install pyspark

pip install img2dataset
```

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

The default filtering conditions in this script are: `LANGUAGE==en`, `WIDTH>=512`, and `HEIGHT>=512`. The resulting metadata should contain 947,191 samples after filtering.

**Notes:**

1. Since OpenCLIP or CLIP is only trained on English text-image data, we need to filter out non-english data via the `LANGUAGE` field to train the Stable Diffusion models for english text-to-image generation.

2. For finetuning based on SD 2.0-base, we prefer images with resolution >= 512x512.

3. To change the filtering conditions, e.g. higher aesthetic, please modified the line of code `df = df.filter(...)` in `laion_filter_metadata.py` accordingly.

### 3. Download and Resize Images 

We will use `img2dataset` to download the image files from URL, resize images, and encode them to local storage.

```shell
output_format="files"

img2dataset --url_list /home/yx/datasets/diffusion/laion_art/laion-art.parquet --input_format "parquet" \
        --url_col "URL" --caption_col "TEXT" \
		--output_format $output_format \
        --output_folder laion-art \
		--processes_count 16 --thread_count 64 --image_size 512 \
        --resize_only_if_bigger=True \
		--resize_mode="keep_ratio" \
		--skip_reencode=True \
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

For more usages of img2dataset, please read the official (API document)[https://github.com/rom1504/img2dataset/tree/main#api].

**Notes**

1. You may change `output_format` to fit your tradeoff between storage space and data loading speed. The options are:
    - files:  saves as a set of subfolder containing pictures (in jpg format by default).
    - webdataset: saves as tars containing pictures, which is compressed and is fast in dataloading.
    - parquet: saves as parquet containing pictures as bytes.
2. For "failed to download" message, please checkout `{output_dir}/0000x_stats.json` for detailed reasons. Here are some solutions to increase the download success rate.
    - To address "certificate verify failed", please replace /your/path/to/python3.7/site-packages/img2dataset/downloader.py by tools/downloaders/downnloader.py to set no certificate context. 
    - Use [DNS resolver](https://github.com/rom1504/img2dataset/tree/main#setting-up-a-high-performance-dns-resolver)
    - TODO: address more download failures in CN network environment.

3. For failed to resize message, you may set `--resize_mode` as "no" to disable resizing, or ajust the reszing parameters.


### 4. Computing text embeddings and latent images with distributed clip-inference and vae-encoder inference. (Optional)

### 5. Convert to trainable data format 

This step is to gather the image path and caption pairs to form the training data.

```
python laion_to_csv.py
```


### 6. Distributed Training

```
sh scripts/run_train_v2_laion.sh
```

It is recommended to run training on a large number of devices (e.g. 128 NPUs), in order to reach a large `batch_size` for GD optimization. A global batch size of **2048** is a reference for producing good training results. 


## Part B. Reproducing SD 2.1-base by finetuning SD 2.0-base on LAION 5B subsets 

The overall pipeline for training on a larger LAION subsets is almost the same as Part A, except for downloading large number of metadata files, image files, and store them efficiently. 


1. Download parquest files

```shell
mkdir laion_2b_en_ae4.5 && cd laion_2b_en_ae4.5
for i in {1..64}; do wget https://huggingface.co/datasets/ChristophSchuhmann/improved_aesthetics_4.5plus/resolve/main/2B-en-4.5_55.parquet; done
cd ..
```

TBC
