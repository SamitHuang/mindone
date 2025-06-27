import os
import numpy as np
import torch
import timm
from safetensors.torch import load_file as safe_load

np.random.seed(42)

def test():
    # /home/andy/models/ViT-SO400M-14-SigLIP-384/open_clip_model.safetensors
    ckpt_path = "/home/hyx/models/timm/ViT-SO400M-14-SigLIP-384/open_clip_model.safetensors"
    model_name = "vit_so400m_patch14_siglip_384"

    img_tensor_path = "./image_tensor.npy"
    if os.path.exists(img_tensor_path):
        input = np.load('./image_tensor.npy')
        print("input loaded successfully")
    else:
        # random tensor
        shape = (1, 3, 384, 384)
        input = np.random.normal(size=shape).astype(np.float32)
        print("no input, created from np.random.normal")

    x = torch.from_numpy(input).to(torch.float32)

    print("start creating model...")
    model = timm.create_model(model_name, pretrained=False)
    model = model.to(torch.float32)
    print("timm created successfully")

    print("start to load state dict")
    state_dict = safe_load(ckpt_path)
    print("loaded state_dict")

    filtered = {}
    prefix = 'visual.trunk.'
    for k, v in state_dict.items():
        if not k.startswith(prefix):
            continue # filter out text, logits etc
        new_k = k[len(prefix):]
        filtered[new_k] = v
    print("finish filtering")

    print("load dict to model")
    missing, unexpected = model.load_state_dict(filtered)
    print("load dict to model finished.")
    model.eval()

    print("start eval")
    with torch.no_grad():
        out = model(x)

    out_np = out.float().cpu().numpy()
    print(out.shape)
    np.save('image_tensor_torch.npy', out_np)
    print(f"result saved to 'image_tensor_torch.npy'")

if __name__ == "__main__":
    test()