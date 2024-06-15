export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MS_ENABLE_NUMA=0
export MS_MEMORY_STATISTIC=1
export MS_DATASET_SINK_QUEUE=4

# enable kbk
export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1
export GLOG_v=2

# hyper-parameters
image_size=512
use_image_num=4
num_frames=17
model_dtype="bf16"
amp_level="O2"
enable_flash_attention="True"
batch_size=2
lr="2e-05"
output_dir=outputs/t2v-dvm_FAbf16_rcM18_rmCast_f$num_frames-$image_size-img$use_image_num-videovae488-$model_dtype-FA$enable_flash_attention-bs$batch_size-t5
msrun --bind_core=True --worker_num=8 --local_worker_num=8 --master_port=9010 --log_dir=test/parallel_logs opensora/train/train_t2v.py \
      --data_path datasets/sharegpt4v_path_cap_64x512x512-vid64.json \
      --video_folder datasets/vid64/videos \
      --text_embed_folder datasets/vid64/t5-len=300 \
      --pretrained models/t2v.ckpt \
      --enable_dvm=True \
    --model LatteT2V-XL/122 \
    --text_encoder_name models/t5-v1_1-xxl \
    --dataset t2v \
    --ae CausalVAEModel_4x8x8 \
    --ae_path LanguageBind/Open-Sora-Plan-v1.1.0 \
    --sample_rate 1 \
    --num_frames $num_frames \
    --max_image_size $image_size \
    --use_recompute True \
    --enable_flash_attention $enable_flash_attention \
    --batch_size=$batch_size \
    --num_parallel_workers 16 \
    --gradient_accumulation_steps=1 \
    --max_train_steps=6000 \
    --start_learning_rate=$lr \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --precision=$model_dtype \
    --amp_level=$amp_level \
    --checkpointing_steps=500 \
    --output_dir=$output_dir \
    --model_max_length 300 \
    --clip_grad True \
    --use_image_num $use_image_num \
    --dataset_sink_mode True \
    --use_img_from_vid \
    --use_parallel True \
    --parallel_mode "data" \
    --mode=0 \
    --num_no_recompute=18 \

    # --enable_tiling \
    # --parallel_mode "optim" \
