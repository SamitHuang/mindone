# export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
# export MS_DATASET_SINK_QUEUE=10

python train.py --config configs/training/mmv2_train.yaml \
    --data_path "../videocomposer/datasets/webvid5" \
    --csv_path "../videocomposer/datasets/webvid5_copy.csv" \
    --output_path "outputs/mmv2_train_webvid5_ms2.3PoC_fa" \
    --enable_flash_attention=True \
    --use_recompute=False \
    --recompute_strategy="down_mm_half" \
    --dataset_sink_mode=True \
    --sink_size=100 \
    --train_steps=40000 \
    --ckpt_save_steps=4000 \
    --train_batch_size 1 \
    --image_size 512 \
