"""layers where the variable names are consistent with pytorch"""
import mindspore.mint as mint
from mindspore import _checkparam as validator
from mindspore.common import dtype as mstype
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import group_norm

'''
class GroupNorm(mint.nn.GroupNorm):
    
    def __init__(self, num_groups, num_channels, eps=1e-05, affine=True, dtype=None):
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
'''
