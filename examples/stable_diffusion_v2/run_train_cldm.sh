export DEVICE_ID=$1
export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE" # debug
# export MS_ASCEND_CHECK_OVERFLOW_MODE=1 # debug

task_name=train_cldm_canny_fill5_infnan_ls65536_wd1e-6
output_dir=outputs/$task_name

rm -rf $output_dir
mkdir -p $output_dir

python train_cldm.py \
    --mode 0 \
    --train_config "configs/train/sd15_controlnet.yaml" \
    --data_path "datasets/fill5" \
    --output_path $output_dir \
    --pretrained_model_path "models/sd_v1.5-d0ab7146_controlnet_init.ckpt" \
    --init_loss_scale 65536 \
    > $output_dir/train.log 2>&1 &

    #--pretrained_model_path "models/control_sd15_canny_ms_static.ckpt" \
