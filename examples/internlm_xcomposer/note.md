
## About
code:
- https://github.com/InternLM/InternLM-XComposer/tree/main
- https://huggingface.co/internlm/internlm-xcomposer2d5-7b/tree/main

## Model

InternLM-XComposer-2.5 (IXC2.5) is built on InternLM-XComposer2 (IXC2) and IXC2-4KHD.
- IXC2: base framework
- IXC2-4KHD: mainly upgrade vision encoder for HD images/videos and dynamic resolution
    - Dynamic Image Partition 
    - Global-Local Format 
    - IXC2.5 update: i) reuse the ViT of 490 × 490 resolution used in IXC2 and further increase its resolution to 560 × 560, so that each sub-image has 400 tokens. ii) a scaled identity strategy to support high-resolution, image is resized and padded to [ph × 560, pw × 560].  

### Vision Encoder: ViT-L/14, 560 × 560 

CLIP ViT-L-14-490 from IXC2 as the vision encoder and further increase its resolution to 560 × 560.

### TODOs
https://huggingface.co/internlm/internlm-xcomposer2d5-clip/blob/main/config.json
- [ ] `CLIPVisionModel` in `transformers.models.clip.modeling_clip.py` 
    - mindnlp: mindnlp/transformers/models/clip/modeling_clip.py 
        - ms2.3.1+910b, graph ok, error:  ; ms2.2.14+gpu, pynative ok.  
    - mindone: missing CLIPImageProcessor


### Text Encoder/Tokenizer
InternLM2 tokenizer, modified from transformers.model.llama.tokenization_llama.LlamaTokenizer

### LLM: InternLM-2 
code: 
- https://github.com/InternLM/InternLM
- https://huggingface.co/internlm/internlm2-chat-7b/tree/main

Key Changes:
1. Consolidate q, k, v matrices
2. Reconfigured maxitrx layout to support **Tensor Parallelism (TP)** transformations 
3. Use Grouped-Query Attention (GQA)

### Glue Layer: Partial LoRA module
Partial LoRA for each linear layer in LLM decoder block;

rank: 256

In code, a MLP vision projector is applied after ViT. When the query contains vision inputs, Partial LoRA (fine-tuned with vision-language data) will be applied in each LLM linear layer.




## Inference

### PT code

Based on `transformers` APIs: AutoModel, model.chat pipeline

``` python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('internlm/internlm-xcomposer2d5-7b', torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval().half()
tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2d5-7b', trust_remote_code=True)
model.tokenizer = tokenizer

query = 'Here are some frames of a video. Describe this video in detail'
image = ['./examples/liuxiang.mp4',]
with torch.autocast(device_type='cuda', dtype=torch.float16):
    response, his = model.chat(tokenizer, query, image, do_sample=False, num_beams=3, use_meta=True)
```

model.chat:
1. [ ] `interleave_wrap_chat`, prepare inputs for LLM
    - [ ] text prompt formatting, tokenize, embedding 
    - [ ] image and video embedding
        - [ ] vit infer
        - [ ] vision project with MLP 
    - concatnate text and vision embeddings, wrap into inputs and mask
2. [ ] `generate`, LLM AR generation
    - re-use `GenerationMixin` in `transformers.generation.utils.py`

## Training
### 1. LVLM Pre-training

LLM frozen. Vision Encoder and Partial LoRA are finetuned, to align visual tokens with LLM.


| Data      | Global BS      | Epochs | Learning rate |  Optimizer |  EMA |
|:-----------|:-------------|:----------|:-----------:|:-----------:|:-----------------:|
| Multiple sources |  4096  | 2      |  2e-4, cosine_decay with warmup,  1% warmupsteps |  ??  |       ??        |

Learning rates:
- Vision Encoder: layer-wise learning rate (LLDR) decay strategy, decay 0.9, to preserve the original knowledge of the vision encoder
- Partial LoRA modules: cosine_decay with warmup, same for each layer

Datasets (Text-Image pairs):
- General sementic: ShareGPT4V-PT, COCO, LAION, etc 
- World knowledge: Concept Data
- Vision capability: RCTW-17, CTW, LSVT (mainly ocr datasets)  

Instruction prompt: *Describe this image briefly/in detail*


### 2. SFT

Jointly train **all** the components with a batch size of 2048 over 4000 steps. (require more memory than pre-training)

Datasets for different tasks:
- Caption: ShareGPT4V, COCO, Nocaps 
- Video: ShareGPT4Video, ActivityNet 
- VQA:
    - OCR QA: TextVQA, OCR-VQA, etc
    - Multi-turn QA: MMDU
    - General: VQAv2 etc 
    - ...
- Conversation: LLaVA-150k, ShareGPT-en&zh, InternLM-Chat

Data sampling: weighted sampling method, the sampling weight of a dataset is related to the number of its samples  

| Data      | Global BS      | Steps | Learning rate |  Optimizer |  EMA |
|:-----------|:-------------|:----------|:-----------:|:-----------:|:-----------------:|
| Multiple sources |  2048  | 4000+   |  1e-5,  each component has its own unique learning strategy |  ??  |       ??        |

Learning rates:
- Vision Encoder: LLDR, decay 0.9
- LLM: fixed learning rate scale factor of 0.2, learn slower


Fine-tune method: LoRA, rank 256


### 3. Preference Alignment for Article Composing task

Preference data collection:
1. sample a prompt *x* from augmented instruction tuning data D
2. generate response *y* with the SFT model, yields 80K prompt-response pairs 
3. use GPT-4o to label (or rank?) the prompt-response pairs: choose (chosen response *yw*)) or reject (rejected response *yl*), and the reason, serve as reward model training data 
4. collect 30K preference data (*x*, *yw*, *yl*) for DPO alignment, refer to the prompt, chosen response and rejected response

DPO alignment
- use the DPO algorithm to update the SFT model 
- minimize the likelihood of dispreferred responses *yl*
- maximize the likelihood of preferred responses *yw*

Fine-tune method: LoRA: rank 256 


### 4. LoRA fine-tuning for Webpage Generation task 

LoRA rank: 512

| Data      | Global BS      | Epochs | Learning rate |  Optimizer |  EMA |
|:-----------|:-------------|:----------|:-----------:|:-----------:|:-----------------:|
| Webpages (image-code pairs) |  512 |  1     |  1e-4, cosine_decay with warmup,  1% warmupsteps |  ??  |       ??        |


