"""
Based on MindCV Inception V3, it creates inception v3 FID variant. 
"""

from typing import Tuple, Union

import mindspore.common.initializer as init
from mindspore import Tensor, nn, ops
from .utils import Download


__all__ = [
    "inception_v3_fid",
]


MS_FID_WEIGHTS_URL = "" #TODO: upload and set url


class Dropout(nn.Dropout):
    def __init__(self, p=0.5, dtype=ms.float32):
        sig = inspect.signature(super().__init__)
        if "keep_prob" in sig.parameters and "p" not in sig.parameters:
            super().__init__(keep_prob=1.0-p, dtype=dtype)
        elif "p" in sig.parameters:
            super().__init__(p=p, dtype=dtype)
        else:
            raise NotImplementedError(
                f"'keep_prob' or 'p' must be the parameter of `mindspore.nn.Dropout`, but got signature of it: {sig}."
            )


class GlobalAvgPooling(nn.Cell):
    """
    GlobalAvgPooling, same as torch.nn.AdaptiveAvgPool2d when output shape is 1
    """

    def __init__(self, keep_dims: bool = False) -> None:
        super().__init__()
        self.keep_dims = keep_dims

    def construct(self, x):
        x = ops.mean(x, axis=(2, 3), keep_dims=self.keep_dims)
        return x


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


class InceptionAux(nn.Cell):
    """Inception module for the aux classifier head"""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        self.avg_pool = nn.AvgPool2d(5, stride=3, pad_mode="valid")
        self.conv0 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.conv1 = BasicConv2d(128, 768, kernel_size=5, pad_mode="valid")
        self.flatten = nn.Flatten()
        self.fc = nn.Dense(in_channels, num_classes)

    def construct(self, x: Tensor) -> Tensor:
        x = self.avg_pool(x)
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class InceptionV3(nn.Cell):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <https://arxiv.org/abs/1512.00567>`_.

    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.

    Args:
        num_classes: number of classification classes. Default: 1000.
        aux_logits: use auxiliary classifier or not. Default: False.
        in_channels: number the channels of the input. Default: 3.
        drop_rate: dropout rate of the layer before main classifier. Default: 0.2.
    """

    def __init__(
        self,
        num_classes: int = 1000,
        aux_logits: bool = True,
        in_channels: int = 3,
        drop_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.aux_logits = aux_logits
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
        if self.aux_logits:
            self.aux = InceptionAux(768, num_classes)
        self.inception7a = InceptionD(768)
        self.inception7b = InceptionE(1280)
        self.inception7c = InceptionE(2048)

        self.pool = GlobalAvgPooling()
        self.dropout = Dropout(p=drop_rate)
        self.num_features = 2048
        self.classifier = nn.Dense(self.num_features, num_classes)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.XavierUniform(), cell.weight.shape, cell.weight.dtype))

    def forward_preaux(self, x: Tensor) -> Tensor:
        x = self.conv1a(x)
        x = self.conv2a(x)
        x = self.conv2b(x)
        x = self.maxpool1(x)
        x = self.conv3b(x)
        x = self.conv4a(x)
        x = self.maxpool2(x)
        x = self.inception5b(x)
        x = self.inception5c(x)
        x = self.inception5d(x)
        x = self.inception6a(x)
        x = self.inception6b(x)
        x = self.inception6c(x)
        x = self.inception6d(x)
        x = self.inception6e(x)
        return x

    def forward_postaux(self, x: Tensor) -> Tensor:
        x = self.inception7a(x)
        x = self.inception7b(x)
        x = self.inception7c(x)
        return x

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.forward_preaux(x)
        x = self.forward_postaux(x)
        return x

    def construct(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = self.forward_preaux(x)
        if self.training and self.aux_logits:
            aux = self.aux(x)
        else:
            aux = None
        x = self.forward_postaux(x)

        x = self.pool(x)
        x = self.dropout(x)
        x = self.classifier(x)

        if self.training and self.aux_logits:
            return x, aux
        return x

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


def _inception_v3(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> InceptionV3:
    """Get InceptionV3 model.
    Refer to the base class `models.InceptionV3` for more details."""
    model = InceptionV3(num_classes=num_classes, aux_logits=True, in_channels=in_channels, **kwargs)

    return model


def inception_v3_fid(ckpt_path=None):
    """Build pretrained Inception model for FID computation

    The Inception model for FID computation uses a different set of weights
    and has a slightly different structure than torchvision's Inception.

    This method first constructs Inception based on MindCV and then patches the
    necessary parts that are different in the FID Inception model.
    """
    inception = _inception_v3(pretrained=False)

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
        
    # load model weights
    if ckpt_path is None:
        assert MS_FID_WEIGHTS_URL, "Either ckpt_path or MS_FID_WEIGHTS_URL MUST be set to load inception v3 model weights for FID calculation." 
        DownLoad().download_url(url=MS_FID_WEIGHTS_URL)
        ckpt_path = os.path.basename(MS_FID_WEIGHTS_URL)

    param_dict = ms.load_checkpoint(ckpt_path)
    ms.load_param_into_net(inception, param_dict)

    return inception

