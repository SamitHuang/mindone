
# operation/graph fusion for dynamic shape
export MS_DEV_ENABLE_KERNEL_PACKET=on

# log level
export GLOG_v=2

output_dir=output/mg1b/$(date +"%Y.%m.%d-%H.%M.%S")

python train.py \
  --config configs/train/stage2_t2iv_256x256.yaml \
  --dataset.sample_n_frames=32 \
  --env.mode 0 \
  --env.jit_level O0 \
  --env.max_device_memory 59GB \
  --model.name llama-1B \
  --tae.pretrained "models/tae.ckpt" \
  --dataset.csv_path datasets/mixkit-100videos/video_caption_train_inflated.csv \
  --dataset.video_folder datasets/mixkit-100videos/mixkit \
  --dataset.text_emb_folder.ul2 datasets/mixkit-100videos/ul2_emb_300 \
  --dataset.text_emb_folder.byt5 datasets/mixkit-100videos/byt5_emb_100 \
  --valid.dataset "" \
  --train.ema "" \
  --train.output_path "$output_dir" \
  --train.save.ckpt_save_interval 500 \
  --train.steps=50000 \

