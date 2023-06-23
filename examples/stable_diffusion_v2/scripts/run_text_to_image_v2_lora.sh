
export GLOG_v=3
export ASCEND_GLOBAL_LOG_LEVEL=3
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export DEVICE_ID=7

export SD_VERSOIN="2.0"


python text_to_image.py \
    --prompt "A drawing of flying dragon" \
    --config configs/v2-inference_lora.yaml \
    --output_path ./output/lora_pokemon/ \
    --seed 42 \
    --n_iter 4 \
    --n_samples 1 \
    --W 512 \
    --H 512 \
    --ddim_steps 50 \
    --use_lora True \
    --ckpt_path output/lora_pokemon/txt2img/ckpt/rank_0 \
    --ckpt_name sd-1_208.ckpt \
