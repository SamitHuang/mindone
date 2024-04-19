export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MS_ENABLE_NUMA=1
export MS_MEMORY_STATISTIC=1

# enable kbk
export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1

export GLOG_v=2

num_frames=64

output_dir=outputs/optim_para_FA_$num_frames

msrun --worker_num=8 --local_worker_num=8 --log_dir=$output_dir/logs train_t2v.py --config configs/train/stdit_256x256x16.yaml \
	--csv_path "../videocomposer/datasets/webvid5/video_caption.csv" \
	--video_folder "../videocomposer/datasets/webvid5" \
	--embed_folder "../videocomposer/datasets/webvid5" \
	--use_parallel True \
	--use_recompute True \
    --image_size=512 \
    --num_frames=$num_frames \
    --enable_flash_attention=True \
    --batch_size=1 \
    --num_parallel_workers=10 \
    --use_ema=False \
    --enable_dvm=True \
    --output_path=$output_dir \
	--parallel_mode "optim"
