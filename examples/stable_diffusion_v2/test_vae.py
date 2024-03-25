from omegaconf import OmegaConf
import mindspore as ms
import numpy as np
import os
from PIL import Image, ImageSequence

from utils import model_utils
from ldm.util import instantiate_from_config

from ldm.data.dataset_vae import ImageDataset
from ldm.models.lpips import LPIPS


def load_model_weights(model, ckpt, verbose=True, prefix_filter=["first_stage_model.", "autoencoder."]):
    sd = ms.load_checkpoint(ckpt)
    
    # filter out vae weights and rename
    all_sd_pnames = list(sd.keys())
    for pname in all_sd_pnames:
        is_vae_param  = False
        for pf in prefix_filter:
            if pname.startswith(pf):
                sd[pname.replace(pf, "")] = sd.pop(pname)
                is_vae_param = True
        if not is_vae_param:
            sd.pop(pname)
    vae_state_dict = sd
    # print(list(sd.keys()))   
    m, u = model_utils.load_param_into_net_with_filter(model, vae_state_dict)
    
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.set_train(False)

    return model


def postprocess(x, trim=True):
    pixels = (x + 1) * 127.5
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    # b, c, h, w -> b h w c
    return np.transpose(pixels, (0, 2, 3, 1))


def test_vae():
    cfg = "configs/train/autoencoder_kl_f8.yaml"
    # ckpt_path = "models/sd_v1.5-d0ab7146.ckpt"
    ckpt_path = "outputs/vae_custom_train/ckpt/vae_kl_f8-e2400.ckpt"
    # ckpt_path = "outputs/vae/ckpt/vae_kl_f8-e1000.ckpt"
    # csv_path = '/home/mindocr/yx/datasets/chinese_art_blip/test/img_txt.csv'
    # image_folder='/home/mindocr/yx/datasets/chinese_art_blip/test'
    csv_path = '/home/mindocr/yx/datasets/chinese_art_blip/train/metadata.csv'
    image_folder='/home/mindocr/yx/datasets/chinese_art_blip/train'
    # csv_path = 'datasets/chinese_art_blip/train/img_txt.csv'
    # image_folder='datasets/chinese_art_blip/train'

    ms.set_context(mode=1)

    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.generator)
    model = load_model_weights(model, ckpt_path)
    # state_dict = ms.load_checkpoint(ckpt_path, model, specify_prefix=["first_stage_model", "autoencoder"])

    model.set_train(False)
    
    ds_config = dict(
            csv_path=csv_path,
            image_folder=image_folder,
            )
    # test source dataset
    ds = ImageDataset(**ds_config)
    sample = ds.__getitem__(0)

    # test construct
    x = ms.Tensor(np.expand_dims(sample, axis=0), dtype=ms.float32)
    recons, posterior_mean, posterior_logvar = model(x)
    # print("mean logvar: ", posterior_mean, posterior_logvar)

    # calc lpips
    perc_loss_fn = LPIPS()
    perc_loss = perc_loss_fn(x, recons)
    print("Recon loss: ", np.abs((x - recons).asnumpy()).mean())
    print("Perception loss: ", perc_loss)

    recon_image = postprocess(recons.asnumpy())[0] 

    Image.fromarray(recon_image).save("tmp_vae_recon.png")

test_vae()

