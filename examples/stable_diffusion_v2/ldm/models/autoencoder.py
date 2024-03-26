# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import functools
import mindspore as ms
from mindspore import nn
from mindspore import ops

from ldm.modules.diffusionmodules.model import Decoder, Encoder
from ldm.models.lpips import LPIPS


class AutoencoderKL(nn.Cell):
    def __init__(
        self,
        ddconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
        use_fp16=False,
        upcast_sigmoid=False,
    ):
        super().__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.image_key = image_key
        self.encoder = Encoder(dtype=self.dtype, upcast_sigmoid=upcast_sigmoid, **ddconfig)
        self.decoder = Decoder(dtype=self.dtype, upcast_sigmoid=upcast_sigmoid, **ddconfig)
        assert ddconfig["double_z"]
        self.quant_conv = nn.Conv2d(
            2 * ddconfig["z_channels"], 2 * embed_dim, 1, pad_mode="valid", has_bias=True
        ).to_float(self.dtype)
        self.post_quant_conv = nn.Conv2d(
            embed_dim, ddconfig["z_channels"], 1, pad_mode="valid", has_bias=True
        ).to_float(self.dtype)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", ms.ops.standard_normal(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.split = ops.Split(axis=1, output_num=2)
        self.exp = ops.Exp()
        self.stdnormal = ops.StandardNormal()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = ms.load_checkpoint(path)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        ms.load_param_into_net(self, sd, strict_load=False)
        print(f"Restored from {path}")

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec
    
    def sample(self, mean, logvar):
        # sample z from gaussian distribution
        logvar = ops.clip_by_value(logvar, -30.0, 20.0)
        std = self.exp(0.5 * logvar)
        z = mean + std * self.stdnormal(mean.shape)
        
        return z

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = self.split(moments)

        # print("D-- mean: ", mean)
        # print("D-- logvar: ", logvar)
        logvar = ops.clip_by_value(logvar, -30.0, 20.0)
        std = self.exp(0.5 * logvar)
        z = mean + std * self.stdnormal(mean.shape)
        
        # TODO: this API change will affact other models using SDv2 VAE!! create another called encode_posterior
        return z, mean, logvar

    def construct(self, input):
        z, posterior_mean, posterior_logvar = self.encode(input)
        recons = self.decode(z)

        return recons, posterior_mean, posterior_logvar


class GeneratorWithLoss(nn.Cell):  
    def __init__(self, 
            autoencoder,
            disc_start=50001,
            kl_weight=1.0e-06,
            disc_weight=0.5,
            disc_factor=1.0,
            perceptual_weight=1.0,
            logvar_init=0.0,
            discriminator=None,
            dtype=ms.float32,
            ):
        super().__init__()

        # build perceptual models for loss compute
        self.autoencoder = autoencoder
        # TODO: set dtype for LPIPS ?
        self.perceptual_loss = LPIPS()  # freeze params inside

        self.l1 = nn.L1Loss(reduction='none')
        # TODO: is self.logvar trainable?
        self.logvar = ms.Parameter(ms.Tensor([logvar_init], dtype=dtype))

        self.disc_start = disc_start
        self.kl_weight = kl_weight
        self.disc_weight = disc_weight
        self.disc_factor = disc_factor
        self.perceptual_weight = perceptual_weight
        
        self.discriminator = discriminator
        # assert discriminator is None, "Discriminator is not supported yet"

    def kl(self, mean, logvar):
        var = ops.exp(logvar)
        kl_loss = 0.5 * ops.sum(
                                ops.pow(mean, 2) + var - 1.0 - logvar,
                                dim=[1, 2, 3],
                                )
        return kl_loss
    
    def loss_function(self, x, recons, mean, logvar, global_step: ms.Tensor=-1, weights: ms.Tensor=None, cond=None):
        bs = x.shape[0] 

        # 2.1 reconstruction loss in pixels
        rec_loss = self.l1(x, recons) 

        # 2.2 perceptual loss
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(x, recons)
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / ops.exp(self.logvar) + self.logvar
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
            mean_weighted_nll_loss = weighted_nll_loss.sum() / bs
            # mean_nll_loss = nll_loss.sum() / bs
        else:
            mean_weighted_nll_loss = nll_loss.sum() / bs
            # mean_nll_loss = mean_weighted_nll_loss 

        # 2.3 kl loss
        kl_loss = self.kl(mean, logvar)
        kl_loss = kl_loss.sum() / bs

        loss = mean_weighted_nll_loss + self.kl_weight * kl_loss

        # 2.4 discriminator loss if enabled
        # g_loss = ms.Tensor(0., dtype=ms.float32)
        # TODO: how to get global_step?
        if global_step >= self.disc_start:
            if (self.discriminator is not None) and (self.disc_factor > 0.):
                # calc gan loss
                if cond is None:
                    logits_fake = self.discriminator(recons)
                else:
                    logits_fake = self.discriminator(ops.concat((recons, cond), dim=1))
                g_loss = -ops.reduce_mean(logits_fake)
                # TODO: do adaptive weighting based on grad
                # d_weight = self.calculate_adaptive_weight(mean_nll_loss, g_loss, last_layer=last_layer)
                d_weight = self.disc_weight
                loss += d_weight * self.disc_factor * g_loss
        # print(f"nll_loss: {mean_weighted_nll_loss.asnumpy():.4f}, kl_loss: {kl_loss.asnumpy():.4f}")

        '''
        split = "train"
        log = {"{}/total_loss".format(split): loss.asnumpy().mean(), 
           "{}/logvar".format(split): self.logvar.value().asnumpy(),
           "{}/kl_loss".format(split): kl_loss.asnumpy().mean(), 
           "{}/nll_loss".format(split): nll_loss.asnumpy().mean(),
           "{}/rec_loss".format(split): rec_loss.asnumpy().mean(),
           # "{}/d_weight".format(split): d_weight.detach(),
           # "{}/disc_factor".format(split): torch.tensor(disc_factor),
           # "{}/g_loss".format(split): g_loss.detach().mean(),
           }
        for k in log:
            print(k.split("/")[1], log[k])
        '''
        # TODO: return more losses

        return loss

    # in graph mode, construct code will run in graph. TODO: in pynative mode, need to add ms.jit decorator
    def construct(self, x: ms.Tensor, global_step: ms.Tensor=-1, weights: ms.Tensor=None, cond=None):
        '''
        x: input image/video, (bs c h w)
        weights: sample weights
        global_step: global training step  
        '''

        # 1. AE forward, get posterior (mean, logvar) and recons
        recons, mean, logvar = self.autoencoder(x)

        # 2. compuate loss
        loss = self.loss_function(x, recons, mean, logvar, global_step, weights, cond)

        return loss


class DiscriminatorWithLoss(nn.Cell):  
    '''
    Training logic:
        For training step i, input data x:
            1. AE generator takes input x, feedforward to get posterior/latent and reconstructed data, and compute ae loss 
            2. AE optimizer updates AE trainable params
            3. D takes the same input x, feed x to AE again **again** to get the new posterior and reconstructions (since AE params has updated), feed x and recons to D, and compute D loss 
            4. D optimizer updates D trainable params 
            --> Go to next training step
        Ref: sd-vae training
    '''
    def __init__(self,
            autoencoder,
            discriminator,
            disc_start=50001,
            disc_factor=1.0,
            disc_loss="hinge",
            ):
        super().__init__()
        self.autoencoder = autoencoder
        self.discriminator = discriminator
        self.disc_start = disc_start
        self.disc_factor = disc_factor

        assert disc_loss in ["hinge", "vanilla"]
        if disc_loss == 'hinge':
            self.disc_loss = self.hinge_loss
        else:
            self.softplus = ops.Softplus()
            self.disc_loss = self.vanilla_d_loss

    def hinge_loss(self, logits_real, logits_fake):
        loss_real = ops.mean(ops.relu(1. - logits_real))
        loss_fake = ops.mean(ops.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss 

    def vanilla_d_loss(self, logits_real, logits_fake):
        d_loss = 0.5 * (ops.mean(self.softplus(-logits_real)) + 
                ops.mean(self.softplus(logits_fake)))
        return d_loss


    def construct(self, x: ms.Tensor, global_step=-1, cond=None):
        '''
        Second pass
        Args:
            x: input image/video, (bs c h w)
            weights: sample weights
        '''

        bs = x.shape[0] 

        # 1. AE forward, get posterior (mean, logvar) and recons
        recons, mean, logvar = ops.stop_gradient(self.autoencoder(x))
        
        # 2. Disc forward to get class prediction on real input and reconstrucions
        if cond is None:
            logits_real = self.discriminator(x)
            logits_fake = self.discriminator(recons)
        else:
            logits_real = self.discriminator(ops.concat((x, cond), dim=1))
            logits_fake = self.discriminator(ops.concat((recons, cond), dim=1))
        
        # TODO: skip previous computation if global step < self.disc_start, to save time
        if global_step >= self.disc_start:
            disc_factor = self.disc_factor 
        else:
            disc_factor = 0.

        d_loss = disc_factor * self.disc_loss(logits_real, logits_fake) 

        # log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
        #        "{}/logits_real".format(split): logits_real.detach().mean(),
        #       "{}/logits_fake".format(split): logits_fake.detach().mean()
        #       }

        return d_loss


def validation_step(input):
    # validate on validatioin set for model selection
    pass


class NLayerDiscriminator(nn.Cell):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> refer to: https://github.com/junyanz/pyms-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False, dtype=ms.float32):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """

        # TODO: check forward consistency!!!
        super().__init__()
        if isinstance(dtype, str):
            if dtype == 'fp32':
                self.dtype = ms.float32
            elif dtype == 'fp16':
                self.dtype = ms.float16
            elif dtype == 'bf16':
                self.dtype = ms.bfloat16
            else:
                raise ValueError
        else:
            self.dtype = dtype

        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        # Fixed
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, pad_mode='pad', padding=padw, has_bias=True).to_float(self.dtype),
                    nn.LeakyReLU(0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, pad_mode='pad', padding=padw, has_bias=use_bias).to_float(self.dtype),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, pad_mode='pad', padding=padw, has_bias=use_bias).to_float(self.dtype),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]

        # output 1 channel prediction map
        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, pad_mode='pad', padding=padw, has_bias=True).to_float(self.dtype)]
        self.main = nn.SequentialCell(sequence)
        self.cast = ops.Cast()

    def construct(self, x):

        y = self.main(x)
        return y
