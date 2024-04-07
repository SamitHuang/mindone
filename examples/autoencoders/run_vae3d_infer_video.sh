python infer.py --model_config configs/causal_vae_f8_t4.yaml \
    --ckpt_path models/causal_vae_488.ckpt \
    --data_path ../videocomposer/datasets/webvid5 \
    --dataset_name video \
    --frame_stride 1 \
    --num_frames 17 \
    --batch_size 1 \
    --output_path samples/causal_vae_opensora_recons \
