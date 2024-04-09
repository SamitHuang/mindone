# enable kernel-by-kernel mode on ms+910b 
export GRAPH_OP_RUN=1

# FIXME: change dtype to bf16 or amp_level to O0, once MS supports conv3d bf16 or fp32 on 910b.
python infer.py --model_config configs/causal_vae_f8_t4.yaml \
    --ckpt_path models/causal_vae_488.ckpt \
    --data_path datasets/mixkit \
    --dataset_name video \
    --amp_level O2 \
    --dtype fp16  \
    --size 512 \
    --crop_size 512 \
    --frame_stride 1 \
    --num_frames 33 \
    --batch_size 1 \
    --output_path samples/causal_vae_recons \
