# AnimateDiff based on MindSpore

## Inference


### Text-to-Video
On 3090 GPU:
```
python infer.py --config configs/prompts/v2/1-ToonYou.yaml --L 16 --H 256 --W 256
```

### Motion LoRA

On 3090 GPU:
```
python infer.py --config configs/prompts/v2/1-ToonYou-MotionLoRA.yaml --L 16 --H 256 --W 256
```
