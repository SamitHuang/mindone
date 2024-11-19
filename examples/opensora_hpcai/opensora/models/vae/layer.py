"""layers where the variable names are consistent with pytorch"""
import mindspore.mint as mint
from mindspore import _checkparam as validator
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import group_norm


class GroupNorm(mint.nn.GroupNorm):
    r"""
    Group Normalization over a mini-batch of inputs.

    Group Normalization is widely used in recurrent neural networks. It applies
    normalization on a mini-batch of inputs for each single training case as described
    in the paper `Group Normalization <https://arxiv.org/pdf/1803.08494.pdf>`_.

    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization, and it performs very stable over a wide
    range of batch size. :math:`\gamma` and :math:`\beta` are trainable scale and shift.
    It can be described using the following formula:

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    where :math:`\gamma` is `weight`, :math:`\beta` is `bias`, and :math:`\epsilon` is `eps`.

    Args:
        num_groups (int): The number of groups to be divided along the channel dimension.
        num_channels (int): The number of input channels.
        eps (float, optional): A value added to the denominator for numerical stability. Default: ``1e-05`` .
        affine (bool, optional): The parameters, such as :math:`\gamma` and :math:`\beta`, are learnable
            when set to ``true`` . Default: ``True`` .
        dtype (:class:`mindspore.dtype`, optional): Dtype of Parameters. Default: ``None`` .

    Inputs:
        - **input** (Tensor) - The input feature with shape :math:`(N, C, *)`, where :math:`*` means, any number of
          additional dimensions.

    Outputs:
        Tensor, the normalized and scaled offset tensor, has the same shape and data type as the `x`.

    Raises:
        TypeError: If `num_groups` or `num_channels` is not an int.
        TypeError: If `eps` is not a float.
        TypeError: If `affine` is not a bool.
        ValueError: If `num_groups` or `num_channels` is less than 1.
        ValueError: If `num_channels` is not divided by `num_groups`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> group_norm_op = ms.mint.nn.GroupNorm(2, 2)
        >>> x = ms.Tensor(np.ones([1, 2, 4, 4], np.float32))
        >>> output = group_norm_op(x)
        >>> print(output)
        [[[[0. 0. 0. 0.]
           [0. 0. 0. 0.]
           [0. 0. 0. 0.]
           [0. 0. 0. 0.]]
          [[0. 0. 0. 0.]
           [0. 0. 0. 0.]
           [0. 0. 0. 0.]
           [0. 0. 0. 0.]]]]
    """

    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, dtype=None):
        """Initialize GroupNorm."""
        super(mint.nn.GroupNorm, self).__init__()
        ms_dtype = mstype.float32 if dtype is None else dtype
        gamma_init = "ones"
        beta_init = "zeros"

        self.num_groups = validator.check_positive_int(num_groups, "num_groups", self.cls_name)
        self.num_channels = validator.check_positive_int(num_channels, "num_channels", self.cls_name)
        if num_channels % num_groups != 0:
            raise ValueError(
                f"For '{self.cls_name}', the 'num_channels' must be divided by 'num_groups', "
                f"but got 'num_channels': {num_channels}, 'num_groups': {num_groups}."
            )
        self.eps = validator.check_value_type("eps", eps, (float,), type(self).__name__)
        self.affine = validator.check_bool(affine, arg_name="affine", prim_name=self.cls_name)

        self.weight = Parameter(
            initializer(gamma_init, self.num_channels, dtype=ms_dtype), name="weight", requires_grad=affine
        )
        self.bias = Parameter(
            initializer(beta_init, self.num_channels, dtype=ms_dtype), name="bias", requires_grad=affine
        )

    def _cal_output(self, x):
        """calculate groupnorm output"""
        return group_norm(x, self.num_groups, self.weight, self.bias, self.eps)

    def extend_repr(self):
        return "num_groups={}, num_channels={}, eps={}, affine={}".format(
            self.num_groups, self.num_channels, self.eps, self.affine
        )

    def construct(self, input):
        output = self._cal_output(input)
        return output
