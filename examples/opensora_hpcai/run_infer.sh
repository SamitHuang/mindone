python scripts/inference.py \
    -c configs/cogvideox_5b-v1-5/inference/sample_t2v.yaml \
    --mode 1 \
    --jit_level "O0" \
    --guidance_scale 1.0 \
    --image_size 768 1360 \
    --num_frame 45 \
    --captions "Disco light leaks disco ball light reflections shaped rectangular and line with motion blur effect." "Cloudy moscow kremlin time lapse" \
    # > infer.log 2>&1 &

    # --ckpt_path outputs/dit_45frames_emaFalse/2024-11-20T10-33-14/ckpt/CogVideoX-5B-v1.5-e2640.ckpt \
