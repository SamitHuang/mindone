"""
Based on MindCV Inception V3, it creates inception v3 FID variant.
"""

from packaging import version
from typing import Tuple, Union

import numpy as np
import mindspore as ms
import mindspore.common.initializer as init
from mindspore import Tensor, nn, ops
from utils import Download


__all__ = [
    "InceptionV3_FID",
    "inception_v3_fid",
]


MS_FID_WEIGHTS_URL = "" #TODO: upload and set url


class BasicConv2d(nn.Cell):
    """A block for conv bn and relu"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple] = 1,
        stride: int = 1,
        padding: int = 0,
        pad_mode: str = "same",
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=padding, pad_mode=pad_mode)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.9997)
        self.relu = nn.ReLU()

    def construct(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InceptionA(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        pool_features: int,
    ) -> None:
        super().__init__()
        self.branch0 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.branch1 = nn.SequentialCell([
            BasicConv2d(in_channels, 48, kernel_size=1),
            BasicConv2d(48, 64, kernel_size=5)
        ])
        self.branch2 = nn.SequentialCell([
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3),
            BasicConv2d(96, 96, kernel_size=3)

        ])
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode="same"),
            BasicConv2d(in_channels, pool_features, kernel_size=1)
        ])

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        branch_pool = self.branch_pool(x)
        out = ops.concat((x0, x1, x2, branch_pool), axis=1)
        return out


class InceptionB(nn.Cell):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch0 = BasicConv2d(in_channels, 384, kernel_size=3, stride=2, pad_mode='valid')
        self.branch1 = nn.SequentialCell([
            BasicConv2d(in_channels, 64, kernel_size=1),
            BasicConv2d(64, 96, kernel_size=3),
            BasicConv2d(96, 96, kernel_size=3, stride=2, pad_mode="valid")

        ])
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        branch_pool = self.branch_pool(x)
        out = ops.concat((x0, x1, branch_pool), axis=1)
        return out


class InceptionC(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        channels_7x7: int,
    ) -> None:
        super().__init__()
        self.branch0 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.branch1 = nn.SequentialCell([
            BasicConv2d(in_channels, channels_7x7, kernel_size=1),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7)),
            BasicConv2d(channels_7x7, 192, kernel_size=(7, 1))
        ])
        self.branch2 = nn.SequentialCell([
            BasicConv2d(in_channels, channels_7x7, kernel_size=1),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1)),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(1, 7)),
            BasicConv2d(channels_7x7, channels_7x7, kernel_size=(7, 1)),
            BasicConv2d(channels_7x7, 192, kernel_size=(1, 7))
        ])
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode="same"),
            BasicConv2d(in_channels, 192, kernel_size=1)
        ])

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        branch_pool = self.branch_pool(x)
        out = ops.concat((x0, x1, x2, branch_pool), axis=1)
        return out


class InceptionD(nn.Cell):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch0 = nn.SequentialCell([
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 320, kernel_size=3, stride=2, pad_mode="valid")
        ])
        self.branch1 = nn.SequentialCell([
            BasicConv2d(in_channels, 192, kernel_size=1),
            BasicConv2d(192, 192, kernel_size=(1, 7)),  # check
            BasicConv2d(192, 192, kernel_size=(7, 1)),
            BasicConv2d(192, 192, kernel_size=3, stride=2, pad_mode="valid")
        ])
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2)

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        branch_pool = self.branch_pool(x)
        out = ops.concat((x0, x1, branch_pool), axis=1)
        return out


class InceptionE(nn.Cell):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch0 = BasicConv2d(in_channels, 320, kernel_size=1)
        self.branch1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.branch1a = BasicConv2d(384, 384, kernel_size=(1, 3))
        self.branch1b = BasicConv2d(384, 384, kernel_size=(3, 1))
        self.branch2 = nn.SequentialCell([
            BasicConv2d(in_channels, 448, kernel_size=1),
            BasicConv2d(448, 384, kernel_size=3)
        ])
        self.branch2a = BasicConv2d(384, 384, kernel_size=(1, 3))
        self.branch2b = BasicConv2d(384, 384, kernel_size=(3, 1))
        self.branch_pool = nn.SequentialCell([
            nn.AvgPool2d(kernel_size=3, pad_mode="same"),
            BasicConv2d(in_channels, 192, kernel_size=1)
        ])

    def construct(self, x: Tensor) -> Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x1 = ops.concat((self.branch1a(x1), self.branch1b(x1)), axis=1)
        x2 = self.branch2(x)
        x2 = ops.concat((self.branch2a(x2), self.branch2b(x2)), axis=1)
        branch_pool = self.branch_pool(x)
        out = ops.concat((x0, x1, x2, branch_pool), axis=1)
        return out

# adaopt mindcv inception for feature extraction in fid
class FIDInceptionA(InceptionA):
    """InceptionA block patched for FID computation"""

    def __init__(self, in_channels, pool_features):
        super(FIDInceptionA, self).__init__(in_channels, pool_features)
        self.branch_pool = BasicConv2d(in_channels, pool_features, kernel_size=1)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        branch_pool = ops.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [x0, x1, x2, branch_pool]
        return ops.cat(outputs, 1)


class FIDInceptionC(InceptionC):
    """InceptionC block patched for FID computation"""

    def __init__(self, in_channels, channels_7x7):
        super(FIDInceptionC, self).__init__(in_channels, channels_7x7)
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        branch_pool = ops.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [x0, x1, x2, branch_pool]
        return ops.concat(outputs, 1)


class FIDInceptionE_1(InceptionE):
    """First InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_1, self).__init__(in_channels)
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x1 = ops.concat((self.branch1a(x1), self.branch1b(x1)), axis=1)
        x2 = self.branch2(x)
        x2 = ops.concat((self.branch2a(x2), self.branch2b(x2)), axis=1)

        branch_pool = ops.avg_pool2d(x, kernel_size=3, stride=1, padding=1, count_include_pad=False)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [x0, x1, x2, branch_pool]
        return ops.concat(outputs, 1)


class FIDInceptionE_2(InceptionE):
    """Second InceptionE block patched for FID computation"""

    def __init__(self, in_channels):
        super(FIDInceptionE_2, self).__init__(in_channels)
        self.branch_pool = BasicConv2d(in_channels, 192, kernel_size=1)

    def construct(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x1 = ops.concat((self.branch1a(x1), self.branch1b(x1)), axis=1)
        x2 = self.branch2(x)
        x2 = ops.concat((self.branch2a(x2), self.branch2b(x2)), axis=1)
        branch_pool = ops.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        outputs = [x0, x1, x2, branch_pool]
        return ops.concat(outputs, 1)


class InceptionV3(nn.Cell):
    """
    Original InceptionV3 network adopted from MindCV
    """
    def __init__(
        self,
        num_classes: int = 1000,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        self.conv1a = BasicConv2d(in_channels, 32, kernel_size=3, stride=2, pad_mode="valid")
        self.conv2a = BasicConv2d(32, 32, kernel_size=3, stride=1, pad_mode="valid")
        self.conv2b = BasicConv2d(32, 64, kernel_size=3, stride=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3b = BasicConv2d(64, 80, kernel_size=1)
        self.conv4a = BasicConv2d(80, 192, kernel_size=3, pad_mode="valid")
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.inception5b = InceptionA(192, pool_features=32)
        self.inception5c = InceptionA(256, pool_features=64)
        self.inception5d = InceptionA(288, pool_features=64)
        self.inception6a = InceptionB(288)
        self.inception6b = InceptionC(768, channels_7x7=128)
        self.inception6c = InceptionC(768, channels_7x7=160)
        self.inception6d = InceptionC(768, channels_7x7=160)
        self.inception6e = InceptionC(768, channels_7x7=192)

        self.inception7a = InceptionD(768)
        self.inception7b = InceptionE(1280)
        self.inception7c = InceptionE(2048)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.XavierUniform(), cell.weight.shape, cell.weight.dtype))


def load_from_ckpt(net, ckpt_path):
    if ckpt_path is None:
        assert MS_FID_WEIGHTS_URL, "Either ckpt_path or MS_FID_WEIGHTS_URL MUST be set to load inception v3 model weights for FID calculation."
        DownLoad().download_url(url=MS_FID_WEIGHTS_URL)
        ckpt_path = os.path.basename(MS_FID_WEIGHTS_URL)

    param_dict = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(net, param_dict)


class InceptionV3_FID(nn.Cell):
    """InceptionV3 for FID variant, returning feature maps."""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=(DEFAULT_BLOCK_INDEX,),
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False,
                 ckpt_path=None,
                 ):
        """Build pretrained InceptionV3 FID

        Parameters
        ----------
        output_blocks : list of int
            Indices of blocks to return features of. Possible values are:
                - 0: corresponds to output of first max pooling
                - 1: corresponds to output of second max pooling
                - 2: corresponds to output which is fed to aux classifier
                - 3: corresponds to output of final average pooling
        resize_input : bool
            If true, bilinearly resizes input to width and height 299 before
            feeding input to model. As the network without fully connected
            layers is fully convolutional, it should be able to handle inputs
            of arbitrary size, so resizing might not be strictly needed
        normalize_input : bool
            If true, scales the input from range (0, 1) to the range the
            pretrained Inception network expects, namely (-1, 1)
        requires_grad : bool
            If true, parameters of the model require gradients. Possibly useful
            for finetuning the network
        """
        super().__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.CellList()

        # define network layers
        inception = InceptionV3() # original Inception V3

        # modify arch for FID
        inception.inception5b = FIDInceptionA(192, pool_features=32)
        inception.inception5c = FIDInceptionA(256, pool_features=64)
        inception.inception5d = FIDInceptionA(288, pool_features=64)
        inception.inception6b = FIDInceptionC(768, channels_7x7=128)
        inception.inception6d = FIDInceptionC(768, channels_7x7=128)
        inception.inception6c = FIDInceptionC(768, channels_7x7=160)
        inception.inception6d = FIDInceptionC(768, channels_7x7=160)
        inception.inception6e = FIDInceptionC(768, channels_7x7=192)
        inception.inception7b = FIDInceptionE_1(1280)
        inception.inception7c = FIDInceptionE_2(2048)

        self.num_features = 2048

        # load weights from pretrained checkpoint
        load_from_ckpt(inception, ckpt_path)

        # organize network layers according to pt definition
        # Block 0: input to maxpool1
        block0 = [
            inception.conv1a,
            inception.conv2a,
            inception.conv2b,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.SequentialCell(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.conv3b,
                inception.conv4a,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.SequentialCell(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.inception5b,
                inception.inception5c,
                inception.inception5d,
                inception.inception6a,
                inception.inception6b,
                inception.inception6c,
                inception.inception6d,
                inception.inception6e,
            ]
            self.blocks.append(nn.SequentialCell(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.inception7a,
                inception.inception7b,
                inception.inception7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.SequentialCell(*block3))

        for param in self.get_parameters():
            param.requires_grad = requires_grad

    def construct(self, x):
        """Get Inception feature maps
        """
        outp = []

        if self.resize_input:
            # TODO: interpolate arg differs in version. ms2.0: size, ms2.0alpha, 1.10, and eailer: sizes
            if version.parse(ms.__version__) >= version.parse('2.0'):
                x = ops.interpolate(x,
                                    size=(299, 299),
                                    mode='bilinear',
                                    align_corners=False)
            else:
                # TODO: this setting (bilinear and half_pixel) does not support CPU.
                x = ops.interpolate(x,
                                    sizes=(299, 299),
                                    mode='bilinear',
                                    coordinate_transformation_mode='half_pixel')


        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp


def inception_v3_fid(dims=2048, ckpt_path=None):
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than original Inception.
    """

    block_idx = InceptionV3_FID.BLOCK_INDEX_BY_DIM[dims]
    net = InceptionV3_FID(output_blocks=[block_idx], ckpt_path=ckpt_path)

    return net


if __name__=="__main__":
    # simple test
    net = inception_v3_fid(ckpt_path='./inception_v3_fid.ckpt')

    bs = 2
    input_size = (bs, 3, 224, 224)
    #dummy_input = ms.Tensor(np.random.rand(*input_size), dtype=ms.float32)
    dummy_input = ms.Tensor(np.ones(input_size)*0.6, dtype=ms.float32)

    y = net(dummy_input)
    for i, feat in enumerate(y):
        print('Output: ', i, feat.shape, feat.sum())
