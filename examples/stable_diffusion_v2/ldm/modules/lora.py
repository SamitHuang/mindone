import mindspore as ms
from mindspore import nn
from mindspore import ops
from mindspore.ops import functional as F
from mindspore.nn.cell import Cell
from mindspore.ops.primitive import Primitive
from mindspore.nn.layer.activation import get_activation
import mindspore.common.initializer as init

from ldm.modules.attention import CrossAttention
from ldm.util import is_old_ms_version

__all__ = ['LoRADenseLayer', 'inject_trainable_lora']


class LoRADenseLayer(nn.Cell):
    '''
    Dense layer with lora injection, used to replace nn.Dense for lora fintuning. Thus the first three a
    Actionvation is not added here since it will be applied in attention layer mostly.
    '''
    def __init__(
        self, in_features, out_features, has_bias=True, rank=4, dropout_p=0.0, scale=1.0, dtype=ms.float32, activation=None,
    ):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}")
        self.rank = rank
        self.scale = scale
        self.dtype = dtype

        # main/orginal linear layer
        self.linear = nn.Dense(in_features, out_features, has_bias=has_bias).to_float(dtype)

        # side-path/LoRA linear layers, the bias for lora matric should be False
        self.lora_down = nn.Dense(in_features, rank, has_bias=False).to_float(dtype)
        self.lora_up = nn.Dense(rank, out_features, has_bias=False).to_float(dtype)

        if is_old_ms_version:
            self.dropout = nn.Dropout(keep_prob=1 - dropout_p)
        else:
            self.dropout = nn.Dropout(p=dropout_p)

        # activation
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        if activation is not None and not isinstance(self.activation, (Cell, Primitive)):
            raise TypeError(f"For '{self.cls_name}', the 'activation' must be str or Cell or Primitive, but got "
                            f"{type(activation).__name__}.")
        self.activation_flag = self.activation is not None

        self.cast = ops.Cast()

        # init
        self._init_weights()

    def _init_weights(self):
        #nn.init.normal_(self.down.weight, std=1 / rank)
        #nn.init.zeros_(self.up.weight)
        self.lora_down.weight.set_data(
                init.initializer(init.Normal(sigma=1.0/self.rank),
                                     self.lora_down.weight.shape, self.lora_down.weight.dtype))
        self.lora_up.weight.set_data(
                init.initializer(init.Zero(),
                                     self.lora_up.weight.shape, self.lora_up.weight.dtype))
        # no need to init the main linear layer since it will be loaded with weights in orignal dense layer.

    def construct(self, x):
        #ori_dtype = ops.dtype(x) # x.dtype
        #x = self.cast(x, self.dtype)

        h_main = self.linear(x)

        z = self.lora_down(x)
        h_lora = self.lora_up(z)

        h = h_main + self.dropout(h_lora) * self.scale

        if self.activation_flag:
            h = self.activation(h)

        #h = self.cast(h, ori_dtype)
        return h


class LowRankDense(nn.Cell):
    '''
    The lora side-path module with low rank matrices
    '''
    def __init__(
        self, in_features, out_features, rank=4, dtype=ms.float32,
        ):
        super().__init__()

        self.down = nn.Dense(in_features, rank, has_bias=False).to_float(dtype)
        self.up = nn.Dense(rank, out_features, has_bias=False).to_float(dtype)

        self.down.weight.set_data(
            init.initializer(init.Normal(sigma=1.0/rank),
                                     self.down.weight.shape, self.down.weight.dtype))

        self.up.weight.set_data(
            init.initializer(init.Zero(),
                                     self.up.weight.shape, self.up.weight.dtype))
    def construct(self, x):

        z = self.down(x)
        h_lora = self.up(z)

        return h_lora


def inject_trainable_lora(net: nn.Cell, target_modules=[CrossAttention], rank=4, dropout_p=0., scale=1.0, use_fp16=False, log_level=1):
    '''
    Currently only support injecting lora to dense layers in attention modules

    In order to find the target layers, currently the attention moduel must have the attribute of to_q, to_k, to_v and to_out[0], each of which correpsonds to a dense layer. to_out correspnds to a SquentialCell consisting of a dense layer and a dropout layer.
    '''
    dtype = ms.float16 if use_fp16 else ms.float32
    ori_net_stat = {}
    ori_net_stat['num_params'] = len(list(net.get_parameters()))

    # find target layers in target moduels, and inject lora to target layers
    catched_attns = {}
    injected_modules = {}
    injected_trainable_params = {}

    # 1. search target modules
    for sc_name, subcell in net.cells_and_names():
        #print('sub cell: ', name, subcell)
        if isinstance(subcell, tuple(target_modules)):
            catched_attns[sc_name] = subcell

            #hier_path = name.split('.')
            #cur = net
            #for submodule_name in hier_path:
            #    cur = getattr(cur, submodule_name)
            #print('===> Cur point to: ', cur)
            #print(subcell.to_q)

    print('Found target modules for lora inject: ', catched_attns)

    if len(catched_attns) == 0:
        print('There is no target modules in net to inject')
        return net

    for sc_name, subcell in catched_attns.items():
        # 2. find target layers to be injected in the module
        target_dense_layers = [subcell.to_q, subcell.to_k, subcell.to_v, subcell.to_out[0]]
        print(f'Target dense layers in the {sc_name}: ', target_dense_layers)

        # 3. create lora dense layers
        new_lora_dense_layers = []
        for i, tar_dense in enumerate(target_dense_layers):
            #print(name, tar_dense)
            if not isinstance(tar_dense, ms.nn.Dense):
                raise ValueError(f'{tar_dense} is NOT a nn.Dense layer')
            has_bias = getattr(tar_dense, 'has_bias')
            in_channels = getattr(tar_dense, 'in_channels')
            out_channels = getattr(tar_dense, 'out_channels')
            #print('in_channels: ', in_channels)
            #print('out_channels: ', out_channels)
            #print('Has bias?: ', has_bias)
            #print('weight: ', tar_dense.weight, tar_dense.weight.data.sum())
            #print('bias: ', tar_dense.bias, tar_dense.bias.sum() if tar_dense.bias is not None else "", '\n')
            #subcell.to_q.weight

            tmp_lora_dense = LoRADenseLayer(
                    in_features=in_channels,
                    out_features=out_channels,
                    has_bias=has_bias,
                    rank=rank,
                    dtype=dtype)

            # copy orignal weight and bias to lora linear (pointing)
            tmp_lora_dense.linear.weight = tar_dense.weight
            if has_bias:
                tmp_lora_dense.linear.bias= tar_dense.bias

            print(f'Create a lora dense layer.', f'Set its linear weights as pretrained weights in {tar_dense.weight.name}')

            new_lora_dense_layers.append(tmp_lora_dense)

        # replace the 4 dense layers with the created lora layers, mount on
        print('Replacing target dense layers with the created lora layers...')
        subcell.to_q = new_lora_dense_layers[0]
        subcell.to_k = new_lora_dense_layers[1]
        subcell.to_v = new_lora_dense_layers[2]
        subcell.to_out[0] = new_lora_dense_layers[3]

        # TODO: don't know why the renaming dows not work in th end trainable_param
        def _update_param_name(param, prefix_module_name):
            # update param name to prefix for lora_up.weight and lora_down.weight
            if prefix_module_name not in param.name:
                param.name = prefix_module_name + '.' + param.name

        for param in subcell.get_parameters():
            # filter to get lora added params by param name
            #print(param)
            if '.lora_down' in param.name or '.lora_up' in param.name or '.linear.' in param.name:
                _update_param_name(param, sc_name)

                if '.lora_down' in param.name or '.lora_up' in param.name:
                    injected_trainable_params[param.name] = param
        # TODO: instead of using fixed list, pick target dense layer by name string then replace it for better extension.
        #lora_attns[name] = subcell # recored
        #print('Attention module after lora injection: ', subcell)

    #print('=> New net after lora injection: ', net)
    #print('\t=> Attn param names: ', '\n'.join([name+'\t'+str(param.requires_grad) for name, param in net.parameters_and_names() if '.to_' in name]))
    print('Parameters in attention layers after lora injection: ', "\n".join([f"{p.name}\t{p}" for p in net.get_parameters() if 'to_' in p.name]))

    new_net_stat = {}
    new_net_stat['num_params'] = len(list(net.get_parameters()))

    print('Ori net stat: ', ori_net_stat)
    print('New net stat: ', new_net_stat)
    assert new_net_stat['num_params'] - ori_net_stat['num_params'] == len(catched_attns) * len(target_dense_layers) * 2, 'Num of parameters should be increased by num_attention_layers * 4 * 2 after injection.'

    injected_modules = catched_attns

    return injected_modules, injected_trainable_params


def save_lora_trainable_params_only(net, ckpt_fp):
    ms.save_checkpoint([{"name":p.name, "data": p} for p in net.trainable_params()], ckpt_fp) # only save lora trainable params only


def load_lora_trainable_params_only(net, lora_ckpt_fp):
    '''
    net should have load orignal pretrained params and injected with lora trainable params. Here we only load the lora trainable params.
    '''
    # TODO: ignore the warning. or don't use the load_checkpoint API. Just manually set parameter values for allora params.
    #lora_ckpt_fp = 'test_lora_tp_after_ft.ckpt'
    param_dict = ms.load_checkpoint(lora_ckpt_fp)
    # TODO: ignore the warning
    net_not_load, ckpt_not_load = ms.load_param_into_net(net, param_dict)
