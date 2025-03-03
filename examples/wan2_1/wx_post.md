# 开箱即用！Wan2.1视频生成模型MindSpore适配开源，昇腾硬件加速加持

在AI视频生成领域，Wan2.1作为最新的视觉生成模型，能够根据文本、图像或其他控制信号生成视频，以其卓越表现备受关注。在VBench评测中，Wan2.1以86.22%的总分，力压Sora、HunyuanVideo等主流模型，摘得桂冠。

MindSpore团队现已完成对Wan2.1的适配，并将其开源至[MindONE]((https://github.com/mindspore-lab/mindone)) 仓库，结合昇腾硬件加速，为开发者提供高效体验。本文将详细介绍如何基于昇思MindSpore和单机Atlas 800T A2，完整实现Wan2.1视频生成的部署流程。

---

## 效果展示：创意一触即发

先别急着看代码，来看看 Wan2.1 的“作品”：

1. **文生视频**：  

https://github.com/user-attachments/assets/f6705d28-7755-447b-a256-6727f66d693b

```text
prompt: A sepia-toned vintage photograph depicting a whimsical bicycle race featuring several dogs wearing goggles and tiny cycling outfits. The canine racers, with determined expressions and blurred motion, pedal miniature bicycles on a dusty road. Spectators in period clothing line the sides, adding to the nostalgic atmosphere. Slightly grainy and blurred, mimicking old photos, with soft side lighting enhancing the warm tones and rustic charm of the scene. 'Bicycle Race' captures this unique moment in a medium shot, focusing on both the racers and the lively crowd.
```

https://github.com/user-attachments/assets/1e1da53a-9112-4fc3-bb8e-b458497c4806

```text
prompt: Film quality, professional quality, rich details. The video begins to show the surface of a pond, and the camera slowly zooms in to a close-up. The water surface begins to bubble, and then a blonde woman is seen coming out of the lotus pond soaked all over, showing the subtle changes in her facial expression, creating a dreamy atmosphere.
```

https://github.com/user-attachments/assets/34e4501f-a207-40bb-bb6c-b162ff6505b0

```text
prompt: Two anthropomorphic cats wearing boxing suits and bright gloves fiercely battled on the boxing ring under the spotlight. Their muscles are tight, displaying the strength and agility of professional boxers. A spotted dog judge stood aside. The animals in the audience around cheered and cheered, adding a lively atmosphere to the competition. The cat's boxing movements are quick and powerful, with its paws tracing blurry trajectories in the air. The screen adopts a dynamic blur effect, close ups, and focuses on the intense confrontation on the boxing ring.
```

https://github.com/user-attachments/assets/aceda253-78a2-4fa5-9edc-83f035c7c2ea

```text
prompt: Sports photography full of dynamism, several motorcycles fiercely compete on the loess flying track, their wheels rolling up the dust in the sky. The motorcyclist is wearing professional racing clothes. The camera uses a high-speed shutter to capture moments, follows from the side and rear, and finally freezes in a close-up of a motorcycle, showcasing its exquisite body lines and powerful mechanical beauty, creating a tense and exciting racing atmosphere. Close up dynamic perspective, perfectly presenting the visual impact of speed and power.
```

https://github.com/user-attachments/assets/c00ca7b8-5e05-4776-8c72-ae19e6bd44f5

```text
prompt: 电影画质，专业质量，丰富细节。一名宇航员身穿太空服，面朝镜头骑着一匹机械马在火星表面驰骋。红色的荒凉地表延伸至远方，点缀着巨大的陨石坑和奇特的岩石结构。机械马的步伐稳健，扬起微弱的尘埃，展现出未来科技与原始探索的完美结合。宇航员手持操控装置，目光坚定，仿佛正在开辟人类的新疆域。背景是深邃的宇宙和蔚蓝的地球，画面既科幻又充满希望，让人不禁畅想未来的星际生活。
```

2. **图生视频**

https://github.com/user-attachments/assets/d37bf480-595e-4a41-95f8-acbc421b7428

```text
prompt: Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.
```
   
猫咪那酷萌的表情绝了! 

这些都是基于MindSpore和昇腾A2硬件跑出来的，效果如何？来动手试试，生成你的专属视频吧！


---

## 快速上手：5分钟玩转Wan2.1

### 环境准备


| mindspore | ascend driver | firmware | cann tookit/kernel |
| :---:     |   :---:       | :---:    | :---:              |
| 2.5.0     |  24.1.0     |7.35.23    |   8.0.RC3.beta1   |

> Ascend cann下载：https://www.hiascend.com/developer/download/community/result

> MindSpore下载：https://www.mindspore.cn/install


### 安装依赖

```
git clone https://github.com/mindspore-lab/mindone
cd mindone/examples/wan2_1

pip install -r requirements.txt
```

### 模型下载

| 模型 |                       下载链接    |   说明   |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| T2V-14B       |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-14B)      🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-14B)          | Supports both 480P and 720P
| I2V-14B-720P  |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P)    🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P)     | Supports 720P
| I2V-14B-480P  |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P)    🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-480P)      | Supports 480P
| T2V-1.3B      |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)     🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-T2V-1.3B)         | Supports 480P

从 Hugging Face 或 ModelScope 下所需的模型，如：

```
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./Wan2.1-T2V-14B
```

### 文本生成视频（T2V）

支持1.3B和14B模型，分辨率可选480P或720P。

- 单卡推理：

```
python generate.py  \
    --task t2v-1.3B \
    --size 832*480 \
    --ckpt_dir ./Wan2.1-T2V-1.3B \
    --sample_shift 8 \
    --sample_guide_scale 6 \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

可自定义prompt，生成480P个性化视频，通过调小`sample_guide_scale`参数增强视频画面质量，或者调大该参数以增强视频-文本匹配程度。

- 多卡加速：

```
msrun --worker_num=4 --local_worker_num=4 generate.py \
    --task t2v-14B \
    --size 1280*720 \
    --ckpt_dir ./Wan2.1-T2V-14B \
    --dit_zero3 --t5_zero3 --ulysses_sp \
    --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
```

可开启序列并行和Zero3模型切片加速720P视频生成

### 图像生成视频（T2V）

支持14B模型，分辨率可选480P或720P。

- 单卡推理：

```
python generate.py \
    --task i2v-14B \
    --size 832*480 \
    --ckpt_dir ./Wan2.1-I2V-14B-480P \
    --image examples/i2v_input.JPG \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```

- 多卡加速：

```
msrun --worker_num=2 --local_worker_num=2 generate.py \
    --task i2v-14B --size 1280*720 \
    --ckpt_dir ./Wan2.1-I2V-14B-720P \
    --dit_zero3 --t5_zero3 --ulysses_sp \
    --image examples/i2v_input.JPG \
    --prompt "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
```


### 性能实测：昇腾硬件加速提升效率

 Atlas 800T A2和MindSpore2.5.0的性能测试结果如下：

|   模型  | 视频尺寸(长x宽x帧数) |     卡数 |  采样步数 | 峰值NPU内存|   生成耗时(s)  |
|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| T2V-1.3B   |      832x480x81 |  1   | 50    |  21GB    |  ~235   |
| T2V-14B   |        1280x720x81 |  1   |  50   |  52.2GB  | ~4650    |
| T2V-14B   |        1280x720x81 |  4   |  50   |  22.6GB  | ~1220   |
| I2V-14B   |        832x480x81 |  1   |  40  |    50GB  | ~1150   |
| I2V-14B   |        1280x720x81 |  4   | 40  |     25GB    | ~1000        |

分析：

- **1.3B模型**：资源占用低，生成速度快，适合轻量应用场景。
- **14B模型**：支持更高分辨率，生成质量更优；多卡并行可显著提升效率，4卡可提速**3.8倍**。


## 结语

想把脑洞变成视频？赶紧去 MindONE 下载代码，动手试试吧！有什么问题，欢迎留言，我们会第一时间帮你解答。

> MindONE开源链接：https://github.com/mindspore-lab/mindone/tree/master/examples/wan2_1



