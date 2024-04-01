import sys
sys.path.append(".")
import mindspore as ms
import numpy as np
from ae.models.modules import  CausalConv3d, ResnetBlock3D


bs, cin, T, H, W = 1, 3, 8, 256, 256
cout = 4  # hidden size, actually 128
x = np.random.normal(size=(bs, cin, T, H, W ))

def test_cconv3d():
    ms.set_context(mode=0)

    conv_in = CausalConv3d(cin, cout, kernel_size=3, stride=1, padding=1)

    xo = conv_in(ms.Tensor(x, dtype=ms.float32))
    
    print("out shape: ", xo.shape)

def _diff_res(ms_val, pt_val):
    abs_diff = np.fabs(ms_val - pt_val)
    # rel_diff[np.fabs(pt_val)]
    # diff_sq = diff**2
    # mse = diff_sq.mean()
    mae = abs_diff.mean()
    max_ae = abs_diff.max()
    return mae, max_ae

def compare_cconv3d(copy_weights=True):
    import torch
    from pt_cconv3d import CausalConv3d as CConv3d_PT

    cc3_pt = CConv3d_PT(cin, cout, kernel_size=3, stride=1, padding=1)
    out_pt = cc3_pt(torch.Tensor(x)) 
    out_pt = out_pt.detach().numpy()
    print("PT: ", out_pt.shape, out_pt.mean(), out_pt.sum())
    
    if copy_weights:
        torch.save({'model_state_dict': cc3_pt.state_dict(),
                    }, "tests/cc3.pth")

        target_data = []
        for k in cc3_pt.state_dict():
            target_data.append({"name": k, "data": ms.Tensor(cc3_pt.state_dict()[k].detach().numpy())})
        ms.save_checkpoint(target_data, "tests/cc3.ckpt")

    cc3_ms = CausalConv3d(cin, cout, kernel_size=3, stride=1, padding=1)

    if copy_weights:
        ms.load_checkpoint("tests/cc3.ckpt", net=cc3_ms)
    
    out_ms = cc3_ms(ms.Tensor(x, dtype=ms.float32))
    out_ms = out_ms.asnumpy()
    print("MS: ", out_ms.shape, out_ms.mean(), out_ms.sum())
    print("Diff: ", _diff_res(out_pt, out_ms))

def compare_nonlinear(npy_fp=None):
    from ae.models.modules import  nonlinearity
    cout = hidden_size = 128
    x = np.random.normal(size=(bs, cout, T, H, W ))

    outms = nonlinearity(ms.Tensor(x, dtype=ms.float32))
    print(outms.sum().asnumpy())

    import torch
    pt_code_path = "/home/mindocr/yx/Open-Sora-Plan/opensora/models/ae/videobase" 
    sys.path.append(pt_code_path)
    from modules.ops import nonlinearity as nl_pt 
    
    outpt = nl_pt(torch.Tensor(x))
    print(outpt.sum().detach().numpy())


def my_gn(x, gamma, beta, groups=32, eps=1e-6):
    batch, channel, height, width = x.shape
    x = np.reshape(x, (batch, groups, -1))

    mean = np.mean(x, axis=2, keepdims=True)
    var = np.var(x, axis=2, keepdims=True)
    # var = ((x - mean)**2).sum() / ((channel * height * width ) / groups)
    std = (var + eps)**0.5
    x = (x - mean) / std
    x = np.reshape(x, (batch, channel, height, width))

    gamma = np.reshape(gamma, (-1, 1, 1))
    beta = np.reshape(beta, (-1, 1, 1))

    output = x * gamma + beta

    return output

def compare_gn(npy_fp=None):
    from ae.models.modules import  nonlinearity, Normalize
    cout = hidden_size = 128
    # x = np.random.normal(size=(bs, cout, T, H, W ))
    x = np.random.uniform(size=(bs, cout, T, H), low=0, high=10)

    import torch
    pt_code_path = "/home/mindocr/yx/Open-Sora-Plan/opensora/models/ae/videobase" 
    sys.path.append(pt_code_path)
    from modules.resnet_block import Normalize as Norm_pt 

    npt = Norm_pt(hidden_size)
    npt.eval()
    outpt = npt(torch.Tensor(x))
    print(outpt.sum().detach().numpy())

    def _convert_ckpt(net=npt, name='gn'):
        torch.save({'model_state_dict': net.state_dict(),
                    }, f"tests/{name}.pth")

        target_data = []
        for k in net.state_dict():
            ms_name = k.replace("weight", "gamma").replace("bias", "beta")
            target_data.append({"name": ms_name, "data": ms.Tensor(net.state_dict()[k].detach().numpy())})
        save_fn = f"tests/{name}.ckpt"
        ms.save_checkpoint(target_data, save_fn)
        return save_fn

    ms_ckpt = _convert_ckpt(npt, 'gn')
    # ms 
    nms = Normalize(hidden_size, extend=False)
    nms.set_train(False)
    # print(nms.parameters_dict().keys())
    sd = ms.load_checkpoint(ms_ckpt)
    ms.load_param_into_net(nms, sd)

    outms = nms(ms.Tensor(x, dtype=ms.float32))
    print(outms.sum().asnumpy())
    
    # numpy
    gamma = npt.state_dict()['weight'].detach().numpy()
    beta = npt.state_dict()['bias'].detach().numpy()
    
    my_out = my_gn(x, gamma, beta)
    print('np out: ', my_out.sum())
    

def test_res3d(npy_fp=None, ckpt_fp=None, backend='ms'):
    if npy_fp is None:
        cout = hidden_size = 128
        x = np.random.normal(size=(bs, hidden_size, T, H, W ))
        np.save("tests/resblock_inp.npy", x)
        print("saved random data")
    else:
        x = np.load(npy_fp)
        cout = hidden_size = x.shape[1]
    
    if backend == 'ms':
        # ms
        rb3_ms = ResnetBlock3D(in_channels=hidden_size, out_channels=hidden_size, dropout=0.)
        rb3_ms.set_train(False)
        if ckpt_fp is not None:
            ms.load_checkpoint(ckpt_fp, net=rb3_ms)

        out_ms = rb3_ms(ms.Tensor(x, dtype=ms.float32))
        out = out_ms.asnumpy()
    else:
        import torch
        pt_code_path = "/home/mindocr/yx/Open-Sora-Plan/opensora/models/ae/videobase" 
        sys.path.append(pt_code_path)
        from modules.resnet_block import ResnetBlock3D as RB3D_PT 

        rb3_pt = RB3D_PT(in_channels=hidden_size, out_channels=hidden_size, dropout=0.)
        rb3_pt.eval()

        out_pt = rb3_pt(torch.Tensor(x)) 
        out = out_pt.detach().numpy()

    print(f"{backend}: ", out.shape, out.mean(), out.sum())


def compare_res3d():
    import torch
    pt_code_path = "/home/mindocr/yx/Open-Sora-Plan/opensora/models/ae/videobase" 
    sys.path.append(pt_code_path)
    from modules.resnet_block import ResnetBlock3D as RB3D_PT 
    
    cout = hidden_size = 128
    x = np.random.normal(size=(bs, hidden_size, T, H, W ))
 
    rb3_pt = RB3D_PT(in_channels=hidden_size, out_channels=hidden_size, dropout=0.)
    rb3_pt.eval()

    out_pt = rb3_pt(torch.Tensor(x)) 
    # out_pt = out_pt.detach().numpy()
    # print("PT: ", out_pt.shape, out_pt.mean(), out_pt.sum())

    torch.save({'model_state_dict': rb3_pt.state_dict(),
                }, "tests/rb3.pth")

    target_data = []
    for k in rb3_pt.state_dict():
        if 'norm' in k:
            ms_name = k.replace(".weight", ".gamma").replace(".bias", ".beta")
        else:
            ms_name = k

        target_data.append({"name": ms_name, "data": ms.Tensor(rb3_pt.state_dict()[k].detach().numpy())})
    ms.save_checkpoint(target_data, "tests/rb3.ckpt")

    # ms
    rb3_ms = ResnetBlock3D(in_channels=hidden_size, out_channels=hidden_size, dropout=0.)
    rb3_ms.set_train(False)
    ms.load_checkpoint("tests/rb3.ckpt", net=rb3_ms)

    out_ms = rb3_ms(ms.Tensor(x, dtype=ms.float32))
    # out_ms = out_ms.asnumpy()
    # print("MS: ", out_ms.shape, out_ms.mean(), out_ms.sum())

    print("PT: ", out_pt.shape, out_pt.sum().detach().numpy())
    print("MS: ", out_ms.shape, out_ms.sum().asnumpy())
    # print("Diff: ", _diff_res(out_pt, out_ms))


if __name__=='__main__':
    # test_ccov3d()
    #print("Checking forward with same weight...")
    #compare_cconv3d(copy_weights=True)
    #print("Checking init...")
    #compare_cconv3d(copy_weights=False)
    # compare_nonlinear()
    # compare_res3d()
    compare_gn()

    # inp = 'tests/resblock_inp.npy'
    # test_res3d(inp, "tests/rb3.pth", "pt")
    # test_res3d(inp, "tests/rb3.ckpt", "ms")
