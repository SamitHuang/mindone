export DEVICE_ID=$1
export MS_ENABLE_REF_MODE=1

# prompt="a painting of a mountain with a lake and trees in the foreground and a mountain in the background"

prompt="a painting of a tree with a mountain in the background and a person standing in the foreground with a snow covered ground" \

task_name=lora_$prompt
output_dir=output/$task_name

mkdir -p $output_dir

python text_to_image.py \
        --prompt $prompt \
        --use_lora True \
        --version "1.5" \
        --lora_ckpt_path $2 \
        --output_path=$output_dir \
