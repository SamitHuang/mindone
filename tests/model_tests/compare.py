import pickle
import numpy as np

import mindspore as ms


# def calc_diff(ms_val, pt_val, eps=1e-8, relax=False):
def calc_diff(ms_val, pt_val, eps=1e-5, relax=False):
    if isinstance(ms_val, ms.Tensor):
        ms_val = ms_val.asnumpy()

    abs_diff = np.fabs(ms_val - pt_val)
    # rel_diff = abs_diff / np.fabs(pt_val) 
    rel_diff = abs_diff / (np.fabs(pt_val) + eps)

    # relax
    if relax:
        # rel_diff = abs_diff / (np.fabs(pt_val))
        tot = np.prod(rel_diff.shape)
        n_nan = np.isnan(rel_diff).sum()
        n_inf = np.isinf(rel_diff).sum()
        print(
            "# values: {}, # nan values: {}, # inf values:{}, (nan+inf)/tot: {}".format(
                tot, n_nan, n_inf, (n_nan + n_inf) / tot
            )
        )
        nan_indicator = np.isnan(rel_diff)
        inf_indicator = np.isinf(rel_diff)

        rel_diff = rel_diff[~(nan_indicator + inf_indicator)]
        abs_diff = abs_diff[~(nan_indicator + inf_indicator)]

    mae = abs_diff.mean()
    mre = rel_diff.mean()
    max_ae = abs_diff.max()
    max_re = rel_diff.max()

    return dict(mae=mae, max_ae=max_ae, mre=mre, max_re=max_re)


def read_pickle_value(pkl_path):
    with open(pkl_path, "rb") as fp:
        data = pickle.load(fp)    
        assert len(data.keys())==1, 'cannot pick more than one value'
        name = list(data.keys())[0]
        pkl_val = data[name]
    return pkl_val


def print_diff(ms_val, gt_data_fp):
    if gt_data_fp.endswith("pkl"):
        pta_val = read_pickle_value(gt_data_fp)
    else:
        pta_val = np.load(gt_data_fp)
    res = calc_diff(ms_val, pta_val, relax=False)
    print(res)
    return res, pta_val


if __name__ == "__main__":
    x = np.ones([1024])
    y = np.ones([1024]) - 0.001
    y[-10:] = np.inf
    print(calc_diff(x, y, relax=True))
