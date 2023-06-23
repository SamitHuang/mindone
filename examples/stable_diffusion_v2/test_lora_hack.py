import os
import mindspore as ms
import numpy as np

from ldm.modules.lora import LoRADenseLayer, LowRankDense, inject_trainable_lora
from ldm.modules.lora import freeze_non_lora_params
from ldm.modules.attention import  BasicTransformerBlock, CrossAttention
from ldm.modules.train.tools import set_random_seed

set_random_seed(42)

class SimpleSubNet(ms.nn.Cell):
    def __init__(self, din=128, dh=128, dtype=ms.float32, use_lora=False):
        super().__init__()
        #self.to_q = nn.Dense(din, dh, has_bias=False).to_float(dtype)
        #self.to_k = nn.Dense(din, dh, has_bias=False).to_float(dtype)
        self.down_block_part1 = BasicTransformerBlock(dim=din, n_heads=2, d_head=dh, dtype=dtype, use_lora=use_lora)

        self.logit = ms.nn.Dense(dh, 1).to_float(dtype)

    def construct(self, x):
        feat = self.down_block_part1(x)
        out = self.logit(feat)
        return out

class SimpleNet(ms.nn.Cell):
    def __init__(self, din=128, dh=128, dtype=ms.float32, use_lora=False):
        super().__init__()
        self.proj = ms.nn.Dense(din, din).to_float(dtype)
        self.encoder =  SimpleSubNet(din, dh, dtype=dtype, use_lora=use_lora)

    def construct(self, x):
        x = self.proj(x)

        return self.encoder(x)

def gen_np_data(bs=1, nd=2, fd=128):
    x = np.zeros([bs, nd, fd])
    for i in range(bs):
        for j in range(nd):
            x[i][j] = np.arange(0, fd, dtype=float) / fd / (j + 1)
    return x


def test_finetune_and_save():
    ms.set_context(mode=0)

    use_fp16 = True
    dtype = ms.float16 if use_fp16 else ms.float32
    rank = 4

    test_data = ms.Tensor(gen_np_data(1, 2, 128), dtype=dtype)

    pretrained_ckpt = 'test_ori_net.ckpt'
    #if not os.path.exists(pretrained_ckpt):
    ori_net = SimpleNet(dtype=dtype, use_lora=False)
    # freeze network
    ori_net.set_train(False)

    ori_net_stat = {}
    ori_net_stat['num_params'] = len(list(ori_net.get_parameters()))

    ori_net_output = ori_net(test_data)
    print("Pretrained net ori output: ", ori_net_output.sum())

    for name, param in ori_net.parameters_and_names():
        param.requires_grad = False
    ms.save_checkpoint(ori_net, pretrained_ckpt)
    print('Done creating pretrained ckpt, please rerun.')
    #exit(1)


    # create lora version
    net = SimpleNet(dtype=dtype, use_lora=True)
    # load preatrained
    param_dict = ms.load_checkpoint(pretrained_ckpt)
    net_not_load, ckpt_not_load = ms.load_param_into_net(net, param_dict)

    # 1. check lora params
    injected_trainable_params = freeze_non_lora_params(net)
    print('Injected lora params: ', injected_trainable_params)
    assert len(net.trainable_params())==len(injected_trainable_params), 'Only lora params can be trainable.'
    assert len(injected_trainable_params)==2*4*2, 'Expecting 16injected lora trainable params, but got {len(injected_trainable_params)}'

    # 2. check foward result consistency
    ## since lora_up.weight are init with all zero. h_lora is alwasy zero before finetuning.
    net_output_after_lora_init = net(test_data)
    print('Outupt after lora injection: ', net_output_after_lora_init.sum())
    assert net_output_after_lora_init.sum()==ori_net_output.sum(), f'net_output_after_lora_init should be the same as ori_net_output'

    #ori_net_stat['dense.linear'] = first_attn.to_q.linear.weight.data.sum()
    #ori_net_stat['dense.lora_down'] = first_attn.to_q.lora_down.weight.data.sum()
    #ori_net_stat['dense.lora_up'] = first_attn.to_q.lora_up.weight.data.sum()

    param_sum = 0
    for p in net.get_parameters():
        param_sum = param_sum + p.data.sum()
    print('Net param sum before ft: ', param_sum)

    new_net_stat = {}
    new_net_stat['num_params'] = len(list(net.get_parameters()))
    assert new_net_stat['num_params'] - ori_net_stat['num_params'] == 16, 'Num of parameters should be increased by num_attention_layers * 4 * 2 after injection.'

    #  check finetune correctness
    def _simple_finetune(net):
        from mindspore.nn import TrainOneStepCell, WithLossCell
        loss = ms.nn.MSELoss()
        optim = ms.nn.SGD(params=net.trainable_params())
        #model = ms.Model(net, loss_fn=loss, optimizer=optim)
        net_with_loss = WithLossCell(net, loss)
        train_network = TrainOneStepCell(net_with_loss, optim)
        train_network.set_train()

        input_data = ms.Tensor(np.random.rand(1, 2, 128), dtype=dtype)
        label = ms.Tensor(np.ones([1, 1]), dtype=dtype)
        print('Finetuning...')
        for i in range(10):
            loss_val = train_network(input_data, label)
            print('loss: ', loss_val)

    _simple_finetune(net)
    net.set_train(False)
    #new_net_stat['dense.linear'] = first_attn.to_q.linear.weight.data.sum()
    #new_net_stat['dense.lora_down'] = first_attn.to_q.lora_down.weight.data.sum()
    #new_net_stat['dense.lora_up'] = first_attn.to_q.lora_up.weight.data.sum()

    # check param change
    print('Ori net stat', ori_net_stat)
    print('New net stat', new_net_stat)
    # On Ascend, this equality check can fail, they have neglectable difference on sum. but CPU is ok.
    #assert new_net_stat['dense.linear'].numpy()== ori_net_stat['dense.linear'].numpy(), 'Not equal: {}, {}'.format(new_net_stat['dense.linear'].numpy(), ori_net_stat['dense.linear'].numpy())
    #assert new_net_stat['dense.lora_down'].value != ori_net_stat['dense.lora_down'].value
    #assert new_net_stat['dense.lora_up'].value != ori_net_stat['dense.lora_up'].value

    # check forward after finetuning
    param_sum = 0
    for p in net.get_parameters():
        param_sum = param_sum + p.data.sum()
    print('Net param sum after ft: ', param_sum)

    output_after_ft = net(test_data)
    print('Input data: ', test_data.sum())
    print('Net outupt after lora ft: ', output_after_ft.sum())
    print(f'\t (Before ft: {net_output_after_lora_init.sum()})')
    #assert output_after_ft.sum()!=net_output_after_lora_init.sum()

    # save
    ms.save_checkpoint([{"name":p.name, "data": p} for p in net.trainable_params()], 'test_lora_tp_after_ft.ckpt') # only save lora trainable params only
    ms.save_checkpoint(net, 'test_lora_net_after_ft.ckpt')


def test_load_and_infer():
    ms.set_context(mode=0)
    use_fp16 = True
    dtype = ms.float16 if use_fp16 else ms.float32
    rank = 4

    pretrained_ckpt = 'test_ori_net.ckpt'
    lora_ft_ckpt = 'test_lora_net_after_ft.ckpt'
    lora_ft_ckpt_part = 'test_lora_tp_after_ft.ckpt'
    load_lora_only = False

    net = SimpleNet(dtype=dtype, use_lora=True)
    net.set_train(False)
    for name, param in net.parameters_and_names():
        param.requires_grad = False

    if not load_lora_only:
        param_dict = ms.load_checkpoint(lora_ft_ckpt)
        net_not_load, ckpt_not_load = ms.load_param_into_net(net, param_dict)
        print('Finish loading lora finetune ckpt: ', net_not_load, ckpt_not_load)
    else:
        # load pretrained
        param_dict = ms.load_checkpoint(pretrained_ckpt)
        net_not_load, ckpt_not_load = ms.load_param_into_net(net, param_dict)
        print('Finish loading pretrained ckpt: ', net_not_load, ckpt_not_load)
        # load lora part
        param_dict_lora = ms.load_checkpoint(lora_ft_ckpt_part)
        net_not_load, ckpt_not_load = ms.load_param_into_net(net, param_dict_lora)
        print('Finish loading finetuned lora part ckpt: ', net_not_load, ckpt_not_load)

    # 1. test forward result consistency
    test_data = ms.Tensor(gen_np_data(1, 2, 128), dtype=dtype)
    #test_data = ms.ops.ones([1, 2, 128], dtype=dtype)*0.66
    net_output = net(test_data)
    print('Input data: ', test_data.sum())
    print("Net forward output: ", net_output.sum())


def test_compare_pt():
    from lora_torch import LoraLinear
    din = 128
    dout = 128
    r = 4
    x = np.random.rand(2, din).astype(np.float32)
    use_fp16 = True
    dtype = ms.float32
    if use_fp16:
        x = x.astype(np.float16)
        dtype = ms.float16

    # torch
    import torch
    with torch.no_grad():
        tnet = LoraLinear(din, dout, r=r)
        tout = tnet(torch.Tensor(x))
    print("torch lora: ", tout.sum())

    # ms
    mnet = LowRankDense(din, dout, rank=r, dtype=dtype)
    mnet.set_train(False)
    #print(list(mnet.get_parameters()))

    # copy weights
    t_param_dict = {}
    for name, param in tnet.named_parameters():
        #print('pt param name, ', name, param.size())
        t_param_dict[name] = param

    for name, param in mnet.parameters_and_names():
        #print('ms param name, ', name, param.shape)
        param.requires_grad = False
        #param.set_data()

        # they have the same param names, linear, lora_up, lora_down
        torch_weight = t_param_dict[name].data
        #print('Find torch weight value: ', torch_weight.shape)

        ms_weight = ms.Tensor(torch_weight.numpy())
        param.set_data(ms_weight)

        print(f'Set ms param {name} to torch weights')

    mout = mnet(ms.Tensor(x))
    print("ms lora: ", mout.sum())

    print("diff: ", mout.sum().numpy() - tout.sum().numpy())


if __name__ == '__main__':
    test_compare_pt()
    #test_finetune_and_save()
    #test_load_and_infer()
