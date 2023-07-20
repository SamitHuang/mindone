# Usage:
# To download one part: `sh laion_download_imgs.sh {part_id}`, e.g. `sh laion_download_imgs.sh 1`
# To download all 64 parts: `sh laion_download_imgs.sh 0`

input_folder=/Volumes/Extreme_SSD/LAION/2b_en_ae4.5_meta_filtered # change to your storage path
output_folder=/Volumes/Extreme_SSD/LAION/2b_en_ae4.5_filtered # change to your storage path
part_id=$1 # if set to 0, it will download all images at a time (requiring more than 30TB storage for saving them). If set to a value in {1..64}, it will only download one part of the whole metadata (requiring around 500GB to save one part)

output_format="files"
#output_format="parquet"
# change it to laion_art if training on laion_art dataset
dataset_name="laion2b_en"
#dataset_name="laion_art"
timeout=15 # default: 10. increase if "The read operation timed out"
#encode_quality=95

if [ "$part_id" -gt 0 ]; then
    input_folder=$input_folder/part_$part_id.parquet
    output_folder=$output_folder/part_$part_id
    echo "Download part $part_id of the dataset"
else
    echo "Download the whole dataset"
fi

if [ "$dataset_name" = "laion2b_en" ]; then
    save_additional_columns='["similarity","hash","punsafe","pwatermark","AESTHETIC_SCORE"]'
else
    save_additional_columns='["similarity","hash","punsafe","pwatermark","aesthetic","LANGUAGE"]'
fi

img2dataset --url_list $input_folder --input_format "parquet" \
        --url_col "URL" --caption_col "TEXT" \
		--output_format $output_format \
        --output_folder  $output_folder \
		--processes_count 16 --thread_count 64 --image_size 512 \
        --resize_only_if_bigger=True \
		--resize_mode="keep_ratio" \
		--skip_reencode=True \
        --timeout $timeout \
        --save_additional_columns $save_additional_columns \
        --number_sample_per_shard 10000 \
		#--enable_wandb True

