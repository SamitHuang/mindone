import numpy as np
import mindspore as ms

def calc_diff(ms_val, pt_val, eps=1e-8):
    
    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()

    rel_diff = abs_diff / (np.fabs(pt_val) + eps)
    mre = rel_diff.mean()
    max_re = rel_diff.max()

    return dict(mae=mae, max_ae=max_ae, mre=mre, max_re=max_re)


def calc_diff_from_npy(ms_val, pt_npy):
    if isinstance(ms_val, ms.Tensor):
        ms_val = ms_val.asnumpy()

    pt_val = np.load(pt_npy)
    out = calc_diff(ms_val, pt_val)

    print(out)

