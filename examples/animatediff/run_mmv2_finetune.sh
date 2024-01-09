export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"

# For best performance, set to a SD1.5 checkpoint trained from image-finetuning in 256x256 resolution 
pretrained_sd=models/stable_diffusion/sd_v1.5-d0ab7146.ckpt

python train.py \
    --data_path=datasets/sdgen_gif_overfit \
    --model_config=configs/stable_diffusion/v1-train-mmv2.yaml \
    --pretrained_model_path=$pretrained_sd \
    --weight_decay=0.01 \
    --image_size=256 --num_frames=16 --frame_stride 1 \
    --dataset_sink_mode=False --epochs=2000 --ckpt_save_interval=1000 --start_learning_rate=0.00001 \
    --train_batch_size=4 --init_loss_scale=65536 --use_lora=False \
    --output_path=tmp_outputs/mmv2_sdgen \
    --warmup_steps=10 --use_ema=False --clip_grad=True \
    --unet_initialize_random=False \
    --image_finetune=False \
    --force_motion_module_amp_O2 True
