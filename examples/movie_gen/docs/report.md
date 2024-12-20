

## Temporal Autoencoder (TAE)

TAE is used to encode the RGB pixel-space videos and images into a spatio-temporally compressed latent space. Specially, the input is compressed by 8x across each spatial dimension H and W, and the temporal dimension T. We follow the framework of Meta Movie Gen [[1](#references)] as below.


TODO: put image here


TAE inflates an image autoencoder by adding 1-D temporal convolution in resnet blocks and attention blocks. Temporal compression is done by injecting temporal downsample and upsample layers.


### Design & Implementation Details

In this section, we explore the design and implementation details that are not illustrated in the Movie Gen paper. For example, how to perform padding and initialization for the Conv 2.5-D layers and how to configure the training frames.

#### SD3.5 VAE as the base image encoder

In TAE, the number of channels of the latent space is 16 (C=16). It can help improve both the reconstruction and the generation performance compared to C=4 used in OpenSora or  SDXL vae.

We choose to use the [VAE]() in Stable Diffusion 3.5 as the image encoder to build TAE for its has the same number of latent channels and can generalize well in image generation. 


#### Conv2.5d implementation

Firstly, we replace the Conv2d in VAE with Conv2.5d, which consists of a 2D spatial convolution followed by a 1D temporal convolution.

For 1D temporal convolution, we set kernel size 3, stride 1, symmetric replicate padding with pading size (1, 1), and input/output channels same as spatial conv. We initialize the kernel weight so as to preserve the spatial features (i.e. preserve image encoding after temporal initialization). Therefore, we propose to use `centric` initilaization as illustrated below.  

```python
w = self.conv_temp.weight
ch = int(w.shape[0])
value = np.zeros(tuple(w.shape))
for i in range(ch):
    value[i, i, 0, 1] = 1
w.set_data(ms.Tensor(value, dtype=ms.float32))
```
#### Temporal Downsampling


Paper: "Temporal downsampling is performed via strided convolution with stride of 2". 

For detailed implementation, we use conv1d of kernel size 3, stride 2, and perform symmetric replicate padding before conv1d. We choose use `centric` initilaization as mentioned in conv2.5 design.

To achieve 8x temporal compression, we apply 3 temporal downsampling layer, each after the spatial downsampling layer of the first 3 levels. 

#### Temporal Upsampling
Paper: "upsampling by nearest-neighbour interpolation followed by convolution"

Our design:
1. nearest-neighbour interpolation along the temporal dimension  
2. conv1d: kernel size 3, stride 1, symmetric replicate padding, and `centric` initialization.

To achieve 8x temporal compression, we apply 3 temporal upsampling layer, each after the spatial upsampling layer of the last 3 levels. 

### Loss

We use reconstruction loss, perceptual loss, KL loss, and outlier penalty loss.

- kl loss weight:  1.0e-06
- perceptual and reconstruction loss weight:  1.0
- outlier penalty loss weight: 1.0
  


```
```



## References
<!--- Guideline: Citation format GB/T 7714 is suggested. -->

[1] The Movie Gen team @ Meta. Movie Gen: A Cast of Media Foundation Models. 2024