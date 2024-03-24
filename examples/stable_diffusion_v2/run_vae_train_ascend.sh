python train_autoencoder.py --base_config configs/train/autoencoder_kl_f8.yaml \
    --mode=0 \
    --output_path="outputs/vae" \
    --csv_path="datasets/chinese_art_blip/train/img_txt.csv" \
    --data_path="datasets/chinese_art_blip/train" \
    --dtype="fp32" \
    --init_loss_scale=1 \
    --epochs=1000 \
    --ckpt_save_interval=100 \
    --batch_size=4 \
    --flip=True \
    --random_crop=True \

    # --scale_lr=False \
    # --base_learning_rate=1e-4 \
