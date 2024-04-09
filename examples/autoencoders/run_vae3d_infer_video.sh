python infer.py --model_config configs/causal_vae_f8_t4.yaml \
    --ckpt_path models/causal_vae_488.ckpt \
    --data_path datasets/mixkit \
    --dataset_name video \
    --size 256 \
    --crop_size 256 \
    --frame_stride 1 \
    --num_frames 17 \
    --batch_size 1 \
    --output_path samples/causal_vae_fp32_recons \
