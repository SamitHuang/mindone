import os
import sys
import mindspore as ms
from mindspore import amp
import numpy as np
sys.path.append(".")
from janus.models.vq_model import VQ_16

np.random.seed(42)

def _diff_res(ms_val, pt_val, eps=1e-8):
    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()

    rel_diff = abs_diff / (np.fabs(pt_val) + eps)
    mre = rel_diff.mean()
    max_re = rel_diff.max()

    return dict(mae=mae, max_ae=max_ae, mre=mre, max_re=max_re)


def set_model_param_dtype(model, dtype=ms.bfloat16, keep_norm_fp32=False):
    if model is not None:
        assert isinstance(model, ms.nn.Cell)

        k_num, c_num = 0, 0
        for _, p in model.parameters_and_names():
            # filter norm/embedding position_ids param
            if keep_norm_fp32 and ("norm" in p.name):
                # print(f"param {p.name} keep {p.dtype}") # disable print
                k_num += 1
            elif "position_ids" in p.name:
                k_num += 1
            else:
                c_num += 1
                p.set_dtype(dtype)

        print(f"Convert `{type(model).__name__}` param to {dtype}, keep/modify num {k_num}/{c_num}.")

    return model

def test_decode(pt_ckpt=None, pt_np=None, dtype=ms.float32):
    # shape = (B, C, H, W) = (1, 8, 24, 24)
    # shape = (B, C, H, W) = (1, 8, 12, 12)
    if pt_np:
        pt_data = np.load(pt_np)
        z = pt_data["quant"]
    else:
        z = np.random.normal(size=(B, C, H, W)).astype(np.float32)

    vq = VQ_16()
    if dtype != ms.float32:
        set_model_param_dtype(vq, dtype=dtype, keep_norm_fp32=False)
    if pt_ckpt:
        vq.load_from_checkpoint(pt_ckpt)
    # 
    if dtype != ms.float32:
        amp.auto_mixed_precision(vq, amp_level="O2", dtype=dtype)


    out = vq.decode(ms.Tensor(z))

    print(out.shape)
    print(out.mean(), out.std())

    if pt_np:
        pt_out = pt_data['dec']
        diff = _diff_res(out.asnumpy(), pt_out)
        print(diff)

    return out.asnumpy()


def test_encode(pt_ckpt=None, amp=False):
    # shape = (B, C, H, W) = (1, 8, 24, 24)
    shape = (B, C, H, W) = (1, 3, 64, 64)
    vq = VQ_16()
    x = np.random.normal(size=(B, C, H, W)).astype(np.float32)
    out = vq.encode(ms.Tensor(x))[0]

    print(out.shape)
    print(out.mean(), out.std())

    return out.asnumpy()



if __name__ == '__main__':
    ms.set_context(mode=1)
    # test_encode()
    test_decode("ckpts/Janus-Pro-1B/pytorch_model.bin", pt_np='tests/vq_dec_io.npz', dtype=ms.bfloat16)

