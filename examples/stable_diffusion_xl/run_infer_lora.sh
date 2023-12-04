export DEVICE_ID=$1
export MS_PYNATIVE_GE=1

base_ckpt_path='models/sd_xl_base_1.0_ms.ckpt'
# ckpt_path="outputs/train_lora_r4_InfNan/2023.11.29-17.26.19/weights/SDXL-base-1.0_15000_lora.ckpt"

ckpt_path="outputs/train_lora_pokemon_r4_InfNan/2023.11.30-16.25.12/weights/SDXL-base-1.0_15000_lora.ckpt"
prompt="datasets/pokemon_blip/test/prompts.txt"
# prompt="a very cute looking pokemon with big eyes"

python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/training/sd_xl_base_finetune_lora_910b.yaml \
  --prompt $prompt \
  --weight $base_ckpt_path,$ckpt_path \
  --save_path 'outputs/pokemon_ft_s15000' \
  --device_target Ascend

  # --num_cols 4 \

  # --prompt "a painting of a tree with a mountain in the background and a person standing in the foreground with a snow covered ground" \
