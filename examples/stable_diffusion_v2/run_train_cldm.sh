export DEVICE_ID=$1

task_name=train_cldm_canny_fill50k_moreFp32_wd1e-3

python train_cldm.py \
    --mode 0 \
    --train_config "configs/train/sd15_controlnet.yaml" \
    --data_path "datasets/fill50k" \
    --output_path "output/$task_name" \
    --pretrained_model_path "models/sd_v1.5-d0ab7146_controlnet_init.ckpt" \
    --init_loss_scale 1048576 \
    #--pretrained_model_path "models/control_sd15_canny_ms_static.ckpt" \
