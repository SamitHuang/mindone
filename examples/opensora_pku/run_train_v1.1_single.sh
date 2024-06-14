# enable kbk
export MS_ENABLE_ACLNN=0
export GRAPH_OP_RUN=0
export GLOG_v=2

# hyper-parameters
image_size=512
use_image_num=4
num_frames=17


# if use global_bf16, set amp_level to 0O
model_dtype="bf16"
amp_level="O2"

enable_flash_attention="True"
batch_size=2
lr="2e-05"
output_dir=outputs/t2v-GE-vaeFp16-ditGBF16-rcM6-f$num_frames-$image_size-img$use_image_num-videovae488-$model_dtype-FA$enable_flash_attention-bs$batch_size-t5

python opensora/train/train_t2v.py \
      --data_path datasets/sharegpt4v_path_cap_64x512x512-vid64.json \
      --video_folder datasets/vid64/videos \
      --text_embed_folder datasets/vid64/t5-len=300 \
      --pretrained models/t2v.ckpt \
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
    --max_train_steps=3000 \
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
    --vae_dtype=fp16 \
    --global_bf16 \
    --mode=0 \

    # --enable_tiling \
    # --parallel_mode "optim" \
