export DEVICE_ID=$1
export MS_PYNATIVE_GE=1
#export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"
export MS_ASCEND_CHECK_OVERFLOW_MODE=1
export PYTHONPATH=$(pwd):$PYTHONPATH

task_name=train_db_lora
save_path=outputs/$task_name

rm -rf $save_path
mkdir -p $save_path

# train
python train_dreambooth.py --config configs/training/sd_xl_base_finetune_dreambooth_lora_910b.yaml --weight models/sd_xl_base_1.0_ms.ckpt --instance_data_path datasets/dog --instance_prompt "a photo of a sks dog" --class_data_path ./dog_class --class_prompt "a photo of a dog" --save_path $save_path \
> $save_path/train.log 2>&1 &

# infer
# python demo/sampling_without_streamlit.py --task txt2img --config configs/training/sd_xl_base_finetune_dreambooth_lora_910b.yaml --weigh models/sd_xl_base_1.0_ms.ckpt $save_path/weights/SDXL-base-1.0_2000_lora.ckpt --prompt "a sks dog swimming in a pool"
