export DEVICE_ID=$1
export MS_PYNATIVE_GE=1
base_ckpt_path='models/sd_xl_base_1.0_ms.ckpt'
ckpt_path="outputs/train_lora_r4_InfNan/2023.11.29-17.26.19/weights/SDXL-base-1.0_15000_lora.ckpt"

python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/training/sd_xl_base_finetune_lora_910b.yaml \
  --num_cols 4 \
  --weight $base_ckpt_path,$ckpt_path \
  --prompt "a painting of a tree with a mountain in the background and a person standing in the foreground with a snow covered ground" \
  --device_target Ascend
