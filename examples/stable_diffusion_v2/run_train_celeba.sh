python train_autoencoder.py \
    --config configs/train/vae_train_config_celeba.yaml \
    --model_config configs/train/autoencoder_kl_f8.yaml \
    --device_target="GPU" \
    --mode=0 \
    --output_path="outputs/vae_celeba_train_ema" \
    --data_path="/home/mindocr/yx/datasets/celeba_hq_256/train" \

    # --scale_lr=False \
    # --base_learning_rate=1e-4 \
