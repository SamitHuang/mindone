# Autoencoders based on MindSpore

This repository provides SoTA image and video autoencoders and their training and inference pipelines.

## Features
- VAE (Image Variational AutoEncoder)
    - [x] KL-reg with GAN loss (SD VAE)
    - [ ] VQ-reg with GAN loss (VQ-GAN)
- Causal 3D Autoencoder
    - [ ] VQ-reg with GAN loss (MagViT)
    - [ ] KL-reg with GAN loss

## Installation

```
pip install -r requirements.txt
```

## Variational Autoencoder (VAE)

### Training

Please download the [lpips_vgg-426bf45c.ckpt](https://download-mindspore.osinfra.cn/toolkits/mindone/autoencoders/lpips_vgg-426bf45c.ckpt) checkpoint and put it in `models/.`.

To lauch training, run
```
python train.py --config configs/training/your_train_receipe.yaml
```
> To run on GPUs, please append  --device_target="GPU"
> To run or debug in pynative mode, please append  --mode=1


For example, to train VAE-kl-f8 model on CelebA-HQ dataset, you can run
```
python train.py --config configs/training/vae_celeba.yaml
```
after setting the `data_path` argument to the dataset path.

Note that you can either set arguments by editing the yaml file, or parsing by CLI (e.g. appending `--data_path=datasets/celeba_hq/train` to the training command). The CLI arguments will overwrite the corresponding values in the base yaml config.

#### Key arguments

- `use_discriminator`: default: False. If True, GAN adversarial training will be applied after `disc_start` steps (defined in model config).

For more arguments, please run `python train.py -h`.

### Evaluation

```
python infer.py \
    --model_config configs/autoencoder_kl_f8.yaml \
    --ckpt_path path/to/checkpoint \
    --data_path path/to/test_data \
    --size 256 \
```
By default, it will save the reconstruction results in `samples/vae_recons` and report the PSNR and SSIM evaluation metrics.

For detailed arguments, please run `python infer.py -h`.

### Results on CelebA-HQ

We split the CelebA-HQ dataset into 24,000 images for training and 6,000 images for testing. After 30 epochs of training, the performance and evaluation results on the test set are reported as follows.


| Model          |   Context   |  Precision         | Local BS x Grad. Accu.  |   Resolution  |  Train T. (ms/step)  |  Train FPS  |   PSNR &#8593    | SSIM   &#8593  |
|:---------------|:---------------|:--------------|:-----------------------:|:----------:|:------------:|:----------------:|:----------------:|:----------------:|
| VAE-kl-f8-ema    |    D910\*x1-MS2.2.10       |      FP32   |      12x1    |    256x256  |    700      |  17.14   |   32    |  0.89    |
| VAE-kl-f8    |    3090x1-MS2.3       |      FP32   |      4x1    |    256x256  | 800      |   5  |    32.37   |  0.90    |


<!-- TODO: attach results
Here are some visualization on the reconstruction results.
-->

## Causal 3D Autoencoder

Coming soon...
