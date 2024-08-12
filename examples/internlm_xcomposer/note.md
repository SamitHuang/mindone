
## About
code:
- https://github.com/InternLM/InternLM-XComposer/tree/main
- https://huggingface.co/internlm/internlm-xcomposer2d5-7b/tree/main

## Model

InternLM-XComposer-2.5 is built on InternLM-XComposer2 (IXC2) and IXC2-4KHD.
- IXC2: base framework
- IXC2-4KHD: mainly upgrade vision encoder for HD images/videos

### Vision Encoder: ViT-L/14 HD


### Text Encoder/Tokenizer
InternLM2 tokenizer, modified from transformers.model.llama.tokenization_llama.LlamaTokenizer

### LLM: InternLM-2 
code: 
- https://github.com/InternLM/InternLM
- https://huggingface.co/internlm/internlm2-7b/tree/main


Key Changes:
1. Consolidate q, k, v matrices
2. Reconfigured maxitrx layout to support **Tensor Parallelism (TP)** transformations 
3. Use Grouped-Query Attention (GQA)


### Glue Layer: Partial LoRA module
Partial LoRa for each linear layer in LLM.


## Pipeline
