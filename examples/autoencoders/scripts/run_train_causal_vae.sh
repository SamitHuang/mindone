# enable kernel-by-kernel mode on ms+910b, require ms>2.3
export GRAPH_OP_RUN=1

python train.py --config configs/training/causal_vae_video.yaml
