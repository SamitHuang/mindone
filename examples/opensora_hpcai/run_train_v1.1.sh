export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# improve data loading performance for distributed training: 1
export MS_ENABLE_NUMA=0
# plot memory usage, feature/model: 1
export MS_MEMORY_STATISTIC=0

export MS_DATASET_SINK_QUEUE=4

# enable kbk: 1
export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1

# log level
export GLOG_v=2
num_frames=8

output_dir=outputs/stdit2_dynamic_shapex$num_frames
#export MS_SUBMODULE_LOG_v="{PIPELINE:1, OPTIMIZER:1, RUNTIME_FRAMEWORK:1}"
#	--pretrained_model_path /home_host/lct/opensora_run/models/OpenSora-STDiT-v2-stage3/opensora_v1.1_stage3.ckpt \

msrun --bind_core=True --master_port=8204 --worker_num=8 --local_worker_num=8 --log_dir=$output_dir  \
	python scripts/train_i2v.py --config configs/opensora-v1-1/train/train.yaml \
	--pretrained_model_path /home_host/lct/opensora_run/models/PixArt-XL-2-1024-MS.ckpt \
	--vae_checkpoint /home_host/lct/opensora_run/models/sd-vae-ft-ema.ckpt \
	--csv_path /home_host/lct/opensora_run/datasets/sora_overfitting_dataset_0410/vcg_200_with_length.csv \
	--video_folder /home_host/lct/opensora_run/datasets/sora_overfitting_dataset_0410/ \
	--text_embed_folder /home_host/lct/opensora_run/datasets/sora_overfitting_dataset_0410/t5_emb_200/video200/ \
  --mode=0 \
  --use_parallel True \
  --num_frames=$num_frames \
  --dataset_sink_mode=False \
  --num_parallel_workers=16 \
  --enable_flash_attention=True \
  --gradient_accumulation_steps=1 \
  --use_ema=False \
  --output_path=$output_dir
#sh clear_env_compile_config.sh
