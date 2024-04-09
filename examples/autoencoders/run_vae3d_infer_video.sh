export GRAPH_OP_RUN=1
python infer.py --model_config configs/causal_vae_f8_t4.yaml \
    --ckpt_path models/causal_vae_488.ckpt \
    --data_path datasets/mixkit \
    --dataset_name video \
    --size 256 \
    --crop_size 256 \
    --frame_stride 1 \
    --num_frames 33 \
    --size 512 \
    --crop_size 512 \
    --batch_size 1 \
    --output_path samples/causal_vae_recons \
