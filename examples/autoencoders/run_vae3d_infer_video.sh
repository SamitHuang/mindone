export GRAPH_OP_RUN=1
python infer.py --model_config configs/causal_vae_f8_t4.yaml \
    --ckpt_path models/causal_vae_488.ckpt \
    --data_path /data3/hyx/datasets/mixkit \
    --dataset_name video \
    --frame_stride 1 \
    --num_frames 33 \
    --size 512 \
    --crop_size 512 \
    --batch_size 1 \
    --output_path samples/causal_vae_fp16_recons \
