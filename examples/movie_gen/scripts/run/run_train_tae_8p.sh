export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

opl=False
output_dir=outputs/train_8p_tae_videoOnly_percLW1_tempInitNorm_opl$opl_fp32


msrun --bind_core=True --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
python scripts/train_tae.py \
    --config configs/tae/train/video_ft.yaml \
    --init_loss_scale 1024. \
    --loss_scaler_type dynamic \
    --amp_level "O0" \
    --dtype fp32 \
    --output_path $output_dir \
    --use_outlier_penalty_loss=$opl \
    --jit_level O0 \
    --mode 0 \
    --epochs 20000 \
    --ckpt_save_interval 50 \
    --image_size 256 \
    --num_frames 16 \
    --use_recompute=False \
    --mixed_image_ratio 0. \
    --use_parallel=True \

    # --csv_path "datasets/ucf101_train.csv" \
    # --video_folder datasets/UCF-101  \
    # --use_recompute=True \
    # --csv_path datasets/mixkit_tiny/sharegpt4v_tiny.csv \
    # --video_folder datasets/mixkit_tiny/video \
