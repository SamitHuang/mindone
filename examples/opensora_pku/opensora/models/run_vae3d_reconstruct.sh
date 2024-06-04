export MS_ENABLE_ACLNN=1
export GRAPH_OP_RUN=1

python test_reconstruct.py \
    --ckpt_path /home_host/yx/mindone/examples/opensora_pku/models/v1.1/vae/causal_vae_3d_v2.ckpt \
    --dataset_name video \
    --size 512 \
    --crop_size 256 \
    --num_frames 33 \
    --dtype bf16 \
    --data_path /home_host/yx/mindone/examples/videocomposer/datasets/webvid5 \

