# HunyuanVideo: A Systematic Framework For Large Video Generation Model

Here we provide an efficient MindSpore implementation of [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), an open-source project that aims to foster large-scale video generation model.

This repository is built on the models and code released by Tencent HunyuanVideo. We are grateful for their exceptional work and generous contribution to open source.


## 📑 Plan

- HunyuanVideo (Text-to-Video Model)
  - [x] Inference
  - [x] Training (SFT)
  - [ ] LoRA fine-tune
  - [x] 3D VAE Inference and Training
  - [ ] Web Demo (Gradio)
  - [ ] Multi-NPU parallel inference
- HunyuanVideo (Image-to-Video Model)
  - [ ] Training support
  - [ ] Inference


## 📜 Requirements

| mindspore | ascend driver | firmware | cann tookit/kernel |
| :---:     |   :---:       | :---:    | :---:              |
| 2.4.1     |  24.1.0     |7.35.23    |   8.0.RC3   |

```
pip install -r requirements.txt
```

## 🧱 Prepare Pretrained Models

Please download the pretrained models and optionally convert them to safetensors format following this [instruction](./ckpts/README.md).

## 📀 Inference

Currently, we support text-to-video generation with text embedding pre-computing. Please refer [Text embedding cache](#text-embedding-cache) to prepare the embedding before running the t2v generation.

``` bash
python sample_video.py \
    --video-size 544 960 \
    --video-length 129 \
    --infer-steps 50 \
    --flow-reverse \
    --seed-type 'fixed' \
    --seed 1 \
    --save-path ./results \
    --dit-weight "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt" \
    --prompt "text-prompt" \
    --text-embed-path /path/to/text_embeddings.npz \
    --vae-tiling=True \
```

Please run `python sample_video.py --help` to see more arguments. An example is provided in `scripts/run_t2v_sample.sh`.


## 🔑 Training

### Dataset Preparation

To prepare the dataset for training HuyuanVideo, please refer to the [dataset format](./hyvideo/dataset/README.md).


### Distributed Training

To train HunyuanVideo (13B) on multiple NPUs, we use ZeRO3 and data parallelism as follows:

```bash
bash scripts/train_t2v_zero3.sh
```


## 3D-VAE Inference and Training

### Video Reconstruction and Evalution

To run video reconstruction using 3D-VAE, please use the following command:

```bash
python hyvideo/rec_video.py \
  --video_path input_video.mp4 \
  --rec_path rec.mp4 \
  --height 360 \
  --width 640 \
  --num_frames 33 \
```

The reconstructed video will be saved under `./samples/`.

To run video reconstruction on a folder of videos, please replace the script with `hyvideo/rec_video_folder.py` and use `--real_video_dir` to parse the video folder path.

To evaluate the reconsturcted videos, you may use the  `hyvideo/eval/scripts/cal_psnr.sh` script.

TODOs:
- [ ] For simplicity, remove `rec_video_folder.py` and allow evaluate a video folder in `rec_video.py` (e.g. `--real_video_dir`), and evaluate PSNR when video reconstruction finished. 

### 3D-VAE Training

coming soon...


## Embedding Cache

### Text embedding cache

To generate text embeddings given an annotation file in JSON format, please use the following command:

```bash
python hyvideo/run_text_encoder.py \
  --text_encoder_path /path/to/ckpt \
  --text_encoder_path_2 /path/to/ckpt \
  --data_file_path /path/to/caption.json \
  --output_path /path/to/text_embed_folder \
```

Please refer to [dataset format](hyvideo/dataset/README.md) to write the json file.  A shell script `scripts/run_text_encoder.sh` is provided as well.

### Video embedding cache


## Acknowledgements

We would like to thank the contributors to the [HunyuanVideo](https://arxiv.org/abs/2412.03603), [SD3](https://huggingface.co/stabilityai/stable-diffusion-3-medium), [FLUX](https://github.com/black-forest-labs/flux), [Llama](https://github.com/meta-llama/llama), [LLaVA](https://github.com/haotian-liu/LLaVA), [diffusers](https://github.com/huggingface/diffusers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.
