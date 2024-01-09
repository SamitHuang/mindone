export MS_ASCEND_CHECK_OVERFLOW_MODE="INFNAN_MODE"

python image_finetune.py \
    --data_path=datasets/cnart_image_overfit --model_config=configs/stable_diffusion/v1-train-mmv2.yaml --pretrained_model_path=models/stable_diffusion/sd_v1.5-d0ab7146.ckpt --weight_decay=0.01 \
    --image_size=256 --num_frames=4 \
    --dataset_sink_mode=False --epochs=1000 --ckpt_save_interval=1000 --start_learning_rate=0.00001 \
    --train_batch_size=1 --init_loss_scale=65536 --use_lora=False --output_path=tmp_outputs/mmv2_overfit_img256frm4 --warmup_steps=10 --use_ema=False --clip_grad=True --unet_initialize_random=False --image_finetune=False
