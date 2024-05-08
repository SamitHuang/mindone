export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# improve data loading performance for distributed training: 1
export MS_ENABLE_NUMA=0
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0

export MS_DATASET_SINK_QUEUE=4

# enable kbk: 1
# export MS_ENABLE_ACLNN=0
# export GRAPH_OP_RUN=0

# log level
export GLOG_v=2

output_dir=outputs/stdit_ge_c18-ms23rc2_ditBF16_256x256x16

msrun --bind_core=True --master_port=8200 --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
	python scripts/train.py --config configs/opensora/train/stdit_256x256x16_ms.yaml \
	--csv_path datasets/sora_overfitting_dataset_0410/vcg_200.csv \
	--video_folder datasets/sora_overfitting_dataset_0410 \
	--text_embed_folder datasets/sora_overfitting_dataset_0410 \
    --use_parallel=True \
    --vae_amp_level=O2 \
    --vae_dtype=fp16 \
    --dtype=bf16 \
    --dataset_sink_mode=True \
    --num_parallel_workers=16 \
    --enable_flash_attention=True \
    --gradient_accumulation_steps=1 \
    --use_ema=False \
    --profile=True \
    --output_path=$output_dir \

