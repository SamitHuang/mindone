export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# improve data loading performance for distributed training: 1
export MS_ENABLE_NUMA=0
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0
export MS_DATASET_SINK_QUEUE=8

# operation/graph fusion for dynamic shape
export MS_DEV_ENABLE_KERNEL_PACKET=on

# log level
export GLOG_v=2

dup=""
output_dir=outputs/debug_fm

# --vae_keep_gn_fp32=False \

# msrun --bind_core=True --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
python scripts/train.py \
--pretrained_model_path="models/OpenSora-STDiT-v3/opensora_stdit_v3.ckpt" \
--mode=0 \
--jit_level O1 \
--max_device_memory 55GB \
--config configs/movie_gen/train/t2i.yaml \
--csv_path datasets/mixkit-100videos/video_caption_train${dup}.csv \
--video_folder datasets/mixkit-100videos/mixkit${dup} \
--text_embed_folder  datasets/mixkit-100videos/t5_emb_300${dup} \
--enable_flash_attention=True \
--gradient_accumulation_steps=1 \
--num_parallel_workers=2 \
--prefetch_size=2 \
--use_ema=True \
--output_path=$output_dir \
--use_recompute=True \
--vae_dtype=fp16 \
--custom_train=True \
--train_steps=8000 --ckpt_save_steps=500 \

# --use_parallel=True \

# --num_parallel_workers=8 \
# --prefetch_size=4 \
# --max_rowsize=256 \
