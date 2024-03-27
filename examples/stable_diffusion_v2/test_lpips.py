import glob
import os

import albumentations
import cv2
import numpy as np
from ldm.models.lpips import LPIPS
from PIL import Image
from torch.utils.data import Dataset

import mindspore as ms

ms.set_context()

perceptron_loss = LPIPS()

vis = True

img_folder = "../videocomposer/demo_video"

size = 384
crop_size = 256

img_paths = []
for postfix in ["png", "jpg", "jpeg", "JPEG"]:
    img_paths.extend(sorted(glob.glob(os.path.join(img_folder, f"*.{postfix}"))))
print(img_paths)


def preprocess(img_path):
    image = Image.open(img_path).convert("RGB")
    image = np.array(image)

    image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
    cropper = albumentations.CenterCrop(height=crop_size, width=crop_size)

    def resize_and_crop(img):
        img = image_rescaler(image=img)["image"]
        img = cropper(image=img)["image"]
        return img

    cropped_image = resize_and_crop(image)
    normed_image = (cropped_image / 127.5 - 1.0).astype(np.float32)
    trans_out = normed_image.transpose((2, 0, 1))  # h w c -> c h w

    return trans_out


print(img_paths[2])
img1 = preprocess(img_paths[2])
img2 = preprocess(img_paths[5])


bs = 1
h = w = crop_size
c = 3

input_batch = np.random.normal(size=(bs, c, h, w)).astype(np.float32)
input_batch[0] = img1

np.save("lp_inp.npz", input_batch)

rand_x = np.random.normal(size=(1, 3, 256, 256)).astype(np.float32)
x1 = input_batch
x2 = x1 * 0.5 + 0.1

x1_ms = ms.Tensor(x1, dtype=ms.float32)
x2_ms = ms.Tensor(x2, dtype=ms.float32)

ms_lp = perceptron_loss(x1_ms, x2_ms)
print(ms_lp)

## torch
import torch
from taming.modules.losses.lpips import LPIPS as PT_LPIPS

pt_perceptual_loss = PT_LPIPS().eval()
pt_score = pt_perceptual_loss(torch.Tensor(x1), torch.Tensor(x2))

print(pt_score)
