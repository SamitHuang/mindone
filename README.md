# MindONE

This repository contains SoTA algorithms, models, and interesting projects in the area of multimodal understanding and content generation.

ONE is short for "ONE for all"

## News
- [2025.03.25] We release MindONE [v0.3.0](https://github.com/mindspore-lab/mindone/releases/tag/v0.3.0). More than 15 SoTA generative models are added, including Flux, CogView4, OpenSora2.0, and HunyuanVideo. Have fun!
- [2025.02.21] We support DeepSeek [Janus-Pro](https://huggingface.co/deepseek-ai/Janus-Pro-7B), a SoTA multimodal understanding and generation model. See [here](examples/janus)
- [2024.11.06] MindONE [v0.2.0](https://github.com/mindspore-lab/mindone/releases/tag/v0.2.0) is released

## Quick tour

To install MindONE v0.3.0, please install [MindSpore 2.5.0](https://www.mindspore.cn/install) and run `pip install mindone`

Alternatively, to install the latest version from the `master` branch, please run.
```
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install -e .
```

We support state-of-the-art diffusion models for generating images, audio, and video. Let's get started using [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium) as an example.

**Hello MindSpore** from **Stable Diffusion 3**!

<div>
<img src="https://github.com/townwish4git/mindone/assets/143256262/8c25ae9a-67b1-436f-abf6-eca36738cd17" alt="sd3" width="512" height="512">
</div>

```py
import mindspore
from mindone.diffusers import StableDiffusion3Pipeline

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    mindspore_dtype=mindspore.float16,
)
prompt = "A cat holding a sign that says 'Hello MindSpore'"
image = pipe(prompt)[0][0]
image.save("sd3.png")
```

### supported models under mindone/examples

| task | model  | infer | fine-tune | pretrain | features  |
| :---   |  :---   |  :---     |  :---     |  :---     |  :--  |
| Text-to-Image | [cogview4](https://github.com/mindspore-lab/mindone/blob/master/examples/cogview) 🔥🔥 | ✅ | ❌ | ❌ | support text to image generation |
| Text/Image-to-Video | [wan2_1](https://github.com/mindspore-lab/mindone/blob/master/examples/wan2_1) 🔥🔥 |  ✅  |  ❌ |  ❌  | support text to video and image to video generation  |
| Text-to-Video | [step_video_t2v](https://github.com/mindspore-lab/mindone/blob/master/examples/step_video_t2v) 🔥🔥 | ✅   | ❌ | ❌  | support text to video generation |
| Text-to-Video | [hpcai open sora](https://github.com/mindspore-lab/mindone/blob/master/examples/opensora_hpcai)      | ✅ | ✅ | ✅ | support v1.0/1.1/1.2 large scale training with dp/sp/dsp/zero, v2.0 inference |
| Text-to-Video | [open sora plan](https://github.com/mindspore-lab/mindone/blob/master/examples/opensora_pku) | ✅ | ✅ | ✅ | support v1.0/1.1/1.2/1.3 large scale training with dp/sp/zero |
| Text-to-Image | [stable diffusion](https://github.com/mindspore-lab/mindone/blob/master/examples/stable_diffusion_v2) | ✅ | ✅ | ✅ | support sd 1.5/2.0/2.1, vanilla fine-tune, lora, dreambooth, text inversion|
| Text-to-Image | [stable diffusion xl](https://github.com/mindspore-lab/mindone/blob/master/examples/stable_diffusion_xl)  | ✅ | ✅ | ✅ | support sai style(stability AI) vanilla fine-tune, lora, dreambooth |
| Class-to-Image | [dit](https://github.com/mindspore-lab/mindone/blob/master/examples/dit)     | ✅  | ✅  | ✅  | support text to image fine-tune |
| Text-to-Image | [fit](https://github.com/mindspore-lab/mindone/blob/master/examples/fit) | ✅ | ✅ | ✅ | support text to image generation in dynamic resolution, fine-tune |
| Text-to-Image | [pixart_sigma](https://github.com/mindspore-lab/mindone/blob/master/examples/pixart_sigma)     | ✅ | ✅ | ✅ | support text to image fine-tune at different aspect ratio |
| Text-to-Image | [flux](https://github.com/mindspore-lab/mindone/blob/master/examples/flux) | ✅ | ✅ | ❌ | support text to image generation, fine-tune  |
| Text-to-Image | [latte](https://github.com/mindspore-lab/mindone/blob/master/examples/latte)     |✅  | ✅ | ✅  | support unconditional text to image fine-tune |
| Video Generation | [animate diff](https://github.com/mindspore-lab/mindone/blob/master/examples/animatediff) | ✅  | ✅  | ✅  | support motion module and lora training |
| Video Generation | [video composer](https://github.com/mindspore-lab/mindone/tree/master/examples/videocomposer)     | ✅  | ✅  | ✅  | support conditional video generation with motion transfer and etc.|
| Text-to-Image | [ip adapter](https://github.com/mindspore-lab/mindone/blob/master/examples/ip_adapter)     | ✅  | ✅  | ✅  | support contronlable text to image generation, fine-tune|
| Text-to-Image | [t2i-adapter](https://github.com/mindspore-lab/mindone/blob/master/examples/t2i_adapter)     | ✅  | ✅  | ✅  | support contraollable text to image generation, fine-tune |
| Image-to-Video | [dynamicrafter](https://github.com/mindspore-lab/mindone/blob/master/examples/dynamicrafter)     | ✅  | ❌ | ❌ | support image to video generation |
| Text-to-Image | [hunyuan_dit](https://github.com/mindspore-lab/mindone/blob/master/examples/hunyuan_dit)     | ✅ | ✅ | ✅ | support text to image fine-tune |
| Text-to-Video | [movie gen](https://github.com/mindspore-lab/mindone/blob/master/examples/moviegen)     | ✅ | ✅ | ✅ | support text to video training with model size up to 30B with dp/sp/zero3 |
| Text/Image-to-3D | [hunyuan3d-1.0](https://github.com/mindspore-lab/mindone/blob/master/examples/hunyuan3d_1)     | ✅ | ✅ | ✅ | support text-to-3D and image-to-3D generation |
| Text-to-Image | [kohya_sd_scripts](https://github.com/mindspore-lab/mindone/blob/master/examples/kohya_sd_scripts) | ✅ | ✅ | ❌ | support text to image generation, fine-tune |
| Video Processing | [magvit](https://github.com/mindspore-lab/mindone/blob/master/examples/magvit) |  ✅  |  ✅  |  ✅  | support video encode and decode|
| Image-to-3D | [instantmesh](https://github.com/mindspore-lab/mindone/blob/master/examples/instantmesh) | ✅  | ✅  | ✅  | support image to 3d generation, fine-tune |
| Text-to-Video | [hunyuanvideo](https://github.com/mindspore-lab/mindone/blob/master/examples/hunyuanvideo) | ✅  | ✅  | ✅  | support text to video generation, fine-tune  |
| Text-to-Image | [story_diffusion](https://github.com/mindspore-lab/mindone/blob/master/examples/story_diffusion) | ✅  | ❌| ❌ | support long-range image generation  |
| Multi-modal | [janus](https://github.com/mindspore-lab/mindone/blob/master/examples/janus) | ✅  | ✅  | ✅  | support multi-modal understanding and generation, fine-tune, t2i/vqa/text |
| Image-to-Video | [svd](https://github.com/mindspore-lab/mindone/blob/master/examples/svd) | ✅  |  ✅ | ✅  | support image to video generation |
| Text-to-3D | [mvdream](https://github.com/mindspore-lab/mindone/blob/master/examples/mvdream) |   ✅ |   ✅ |   ✅ | support text to 3d generation, fine-tune  |
| Image-to-3D | [sv3d](https://github.com/mindspore-lab/mindone/blob/master/examples/sv3d) |   ✅ |   ✅ |   ✅ | support image to 3d generation, fine-tune |
| Video Generation | [hunyuanvideo-i2v](https://github.com/mindspore-lab/mindone/blob/master/examples/hunyuanvideo-i2v) | Yes | Yes | Yes | support image to video generation |
| Text-to-Video | [t2v_turbo](https://github.com/mindspore-lab/mindone/blob/master/examples/t2v_turbo) |   ✅ |   ✅ |   ✅ | support text to video generation, fine-tune |
| Video Enhancement | [venhancer](https://github.com/mindspore-lab/mindone/blob/master/examples/venhancer) |  ✅  | ❌ | ❌ | support video to video enhancement |
| Image-Text-to-Text | [qwen2_vl](https://github.com/mindspore-lab/mindone/blob/master/examples/qwen2_vl) | Yes | Yes | Yes | support multi-modal understanding  |



<!-- TODO: add models in PR, emu3, var, etc -->

###  run hf diffusers on mindspore
mindone diffusers is under active development, most tasks were tested with mindspore 2.5.0 and ascend A2 machines.

| component  |  features  
| :---   |  :--  
| [pipeline](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/pipelines) | support text2image,text2video,text2audio tasks 30+
| [models](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/models) | support audoencoder & transformers base models same as hf diffusers
| [schedulers](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/schedulers) | support ddpm & dpm solver 10+ schedulers same as hf diffusers
