import argparse
import mindspore as ms
from safetensors import safe_open
import torch

def convert(pt_ckpt, target_fp):
    if pt_ckpt.endswith('.pth'):
        # state_dict = torch.load(pt_ckpt, map_location="CPU")['model_state_dict']
        state_dict = torch.load(pt_ckpt)['model_state_dict']
    else:
        state_dict = {}
        with safe_open(pt_ckpt, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    target_data = []
    for k in state_dict:
        if '.' not in k:
            # only for GroupNorm
            ms_name = k.replace("weight", "gamma").replace("bias", "beta")
        else:
            if 'norm' in k:
                ms_name = k.replace(".weight", ".gamma").replace(".bias", ".beta")
            else:
                ms_name = k
        target_data.append({"name": ms_name, "data": ms.Tensor(state_dict[k].detach().numpy())})

    save_fn = pt_ckpt.replace(".pth", ".ckpt")
    ms.save_checkpoint(target_data, target_fp)

    return save_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=None, help="path to torch checkpoint path")
    parser.add_argument(
        "--target", type=str, default="models/vae3d.ckpt", help="target file path to save the converted checkpoint"
    )
    args = parser.parse_args()

    convert(args.src, args.target)
