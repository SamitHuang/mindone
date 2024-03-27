python infer_vae.py \
    --data_path /home/mindocr/yx/datasets/celeba_hq_256/test \
    --size 256 \
    --crop_size 256 \
    --batch_size 10 \
    --mode 0 \
    --ckpt_path outputs/vae_celeba_train/ckpt/vae_kl_f8-e22.ckpt \
    --output_path samples/vae_recons_e22 \

    # --measure_loss=True \

    #--ckpt_path models/sd_v1.5-d0ab7146.ckpt \

