rm -rf EXCEPTION*
rm -rf rank_*
rm -rf kernel_meta
rm -rf *.json
rm -rf output
rm -rf log*
# rm -rf outputs
rm -rf *.log

# export PYTHONPATH="/home/yx/mindone:/home/mikecheung/gitlocal/mindone/examples/moviegen:$PYTHONPATH"
export PYTHONPATH="/home_host/yx/mindone:$PYTHONPATH"

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

output_dir=outputs/dit_45frames_emaFalse

msrun --master_port=8201 --worker_num=8 --local_worker_num=8 --log_dir=$output_dir scripts/train.py \
    --config configs/cogvideox_5b-v1-5/train/train_t2v.yaml \
    --csv_path ../videocomposer/datasets/webvid5/video_caption.csv \
    --video_folder ../videocomposer/datasets/webvid5/ \
    --text_embed_folder ../videocomposer/datasets/webvid5_224 \
    --vae_latent_folder ../videocomposer/datasets/webvid5_768_1360 \
    --use_parallel True \
    --zero_stage 2 \
    --jit_level "O1" \
    --num_frames 45 \
    --num_latent_frames 12  \
    --output_path=$output_dir \
    --use_ema False \
    --enable_sequence_parallelism True \
    --sequence_parallel_shards 8


# try with 12
