python infer_vae.py \
    --data_path /home/mindocr/yx/datasets/celeba_hq_256/small_test \
    --output_path samples/vae_recons_e13 \
    --size 256 \
    --crop_size 256 \
    --mode 1 \
    --ckpt_path outputs/vae_celeba_train/ckpt/vae_kl_f8-e13.ckpt \

    #--ckpt_path models/sd_v1.5-d0ab7146.ckpt \

