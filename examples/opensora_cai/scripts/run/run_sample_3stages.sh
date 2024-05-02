python scripts/infer_t5.py--config  configs/opensora/inference/stdit_512x512x64.yaml --output_path  samples/t5_embed.npz

python scripts/inference.py --config configs/opensora/inference/stdit_512x512x64.yaml --embed_path samples/t5_embed.npz --use_vae_decode=False \
    --dtype=fp16 \
    --amp_level="O2" \
    --enable_flash_attention=True \
    --ckpt_path /path/to/stdit.ckpt

python scripts/infer_vae_decode.py --fps=12 --latent_path  \
     samples/denoised_latent_00.npy \
     samples/denoised_latent_01.npy \
     samples/denoised_latent_02.npy \
     samples/denoised_latent_03.npy \
     samples/denoised_latent_04.npy \