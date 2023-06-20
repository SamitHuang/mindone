"""
Convert pytorch checkpoint to mindspore checkpoint for inception v3.
The converted model `inception_v3_fid.ckpt` will be saved in the same directory as this file belonging to.

Require to install both pytorch and mindspore before running this script.
"""
import os
import torch
import mindspore as ms
from mindspore.train.serialization import save_checkpoint
from utils import Download, _DEFAULT_DOWNLOAD_ROOT
from tqdm import tqdm

PT_FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'  # noqa: E501

def torch_to_mindspore(pt_ckpt, save=True, save_fp='./inception_v3_fid.ckpt'):

    state_dict = torch.load(pt_ckpt, map_location=torch.device('cpu'))

    ms_params = []
    for k, v in tqdm(state_dict.items()):
        if 'fc' in k:
            k = k.replace('fc', 'classifier')
        if 'num_batches_tracked' in k:
            continue
        if 'Conv2d_' in k:
            k = k.replace('Conv2d_', 'conv')
            if '_3x3' in k:
                k = k.replace('_3x3', '')
        if 'running_mean' in k:
            k = k.replace('running_mean', 'moving_mean')
        if 'running_var' in k:
            k = k.replace('running_var', 'moving_variance')
        if 'bn.weight' in k:
            k = k.replace('bn.weight', 'bn.gamma')
        if 'bn.bias' in k:
            k = k.replace('bn.bias', 'bn.beta')
        if 'conv3b_1x1' in k:
            k = k.replace('conv3b_1x1', 'conv3b')
        if 'Mixed_5' in k:
            k = k.replace('Mixed_5', 'inception5')
            if 'branch1x1' in k:
                k = k.replace('branch1x1', 'branch0')
            if 'branch5x5_1' in k:
                k = k.replace('branch5x5_1', 'branch1.0')
            if 'branch5x5_2' in k:
                k = k.replace('branch5x5_2', 'branch1.1')
            if 'branch3x3dbl_1' in k:
                k = k.replace('branch3x3dbl_1', 'branch2.0')
            if 'branch3x3dbl_2' in k:
                k = k.replace('branch3x3dbl_2', 'branch2.1')
            if 'branch3x3dbl_3' in k:
                k = k.replace('branch3x3dbl_3', 'branch2.2')
        if 'Mixed_6a' in k:
            k = k.replace('Mixed_6', 'inception6')
            if 'branch3x3' in k:
                k = k.replace('branch3x3', 'branch0')
            if 'branch3x3dbl_1' in k:
                k = k.replace('branch3x3dbl_1', 'branch1.0')
            if 'branch3x3dbl_2' in k:
                k = k.replace('branch3x3dbl_2', 'branch1.1')
            if 'branch3x3dbl_3' in k:
                k = k.replace('branch3x3dbl_3', 'branch1.2')
            if 'branch0dbl_1' in k:
                k = k.replace('branch0dbl_1', 'branch1.0')
            if 'branch0dbl_2' in k:
                k = k.replace('branch0dbl_2', 'branch1.1')
            if 'branch0dbl_3' in k:
                k = k.replace('branch0dbl_3', 'branch1.2')

        if 'Mixed_6b' in k or 'Mixed_6c' in k or 'Mixed_6d' in k or 'Mixed_6e' in k:
            k = k.replace('Mixed_6', 'inception6')
            if 'branch1x1' in k:
                k = k.replace('branch1x1', 'branch0')
            if 'branch7x7_1' in k:
                k = k.replace('branch7x7_1', 'branch1.0')
            if 'branch7x7_2' in k:
                k = k.replace('branch7x7_2', 'branch1.1')
            if 'branch7x7_3' in k:
                k = k.replace('branch7x7_3', 'branch1.2')
            if 'branch7x7dbl_1' in k:
                k = k.replace('branch7x7dbl_1', 'branch2.0')
            if 'branch7x7dbl_2' in k:
                k = k.replace('branch7x7dbl_2', 'branch2.1')
            if 'branch7x7dbl_3' in k:
                k = k.replace('branch7x7dbl_3', 'branch2.2')
            if 'branch7x7dbl_4' in k:
                k = k.replace('branch7x7dbl_4', 'branch2.3')
            if 'branch7x7dbl_5' in k:
                k = k.replace('branch7x7dbl_5', 'branch2.4')

        if 'Mixed_7a' in k:
            k = k.replace('Mixed_7', 'inception7')
            if 'branch3x3_1' in k:
                k = k.replace('branch3x3_1', 'branch0.0')
            if 'branch3x3_2' in k:
                k = k.replace('branch3x3_2', 'branch0.1')
            if 'branch7x7x3_1' in k:
                k = k.replace('branch7x7x3_1', 'branch1.0')
            if 'branch7x7x3_2' in k:
                k = k.replace('branch7x7x3_2', 'branch1.1')
            if 'branch7x7x3_3' in k:
                k = k.replace('branch7x7x3_3', 'branch1.2')
            if 'branch7x7x3_4' in k:
                k = k.replace('branch7x7x3_4', 'branch1.3')

        if 'Mixed_7b' in k or 'Mixed_7c' in k:
            k = k.replace('Mixed_7', 'inception7')
            if 'branch1x1' in k:
                k = k.replace('branch1x1', 'branch0')
            if 'branch3x3_1' in k:
                k = k.replace('branch3x3_1', 'branch1')
            if 'branch3x3_2a' in k:
                k = k.replace('branch3x3_2a', 'branch1a')
            if 'branch3x3_2b' in k:
                k = k.replace('branch3x3_2b', 'branch1b')
            if 'branch3x3dbl_1' in k:
                k = k.replace('branch3x3dbl_1', 'branch2.0')
            if 'branch3x3dbl_2' in k:
                k = k.replace('branch3x3dbl_2', 'branch2.1')
            if 'branch3x3dbl_3a' in k:
                k = k.replace('branch3x3dbl_3a', 'branch2a')
            if 'branch3x3dbl_3b' in k:
                k = k.replace('branch3x3dbl_3b', 'branch2b')
        ms_params.append({'name': k, 'data': ms.Tensor(v.numpy())})

    if save:
        save_checkpoint(ms_params, save_fp)

    return ms_params

def main():
    # download torch checkpoint
    Download().download_url(url=PT_FID_WEIGHTS_URL)
    filename = os.path.basename(PT_FID_WEIGHTS_URL)
    pt_fp = os.path.join(_DEFAULT_DOWNLOAD_ROOT, filename)

    # convert to ms checkpoint
    __dir__ = os.path.dirname(os.path.abspath(__file__))
    ckpt_save_fp = os.path.join(__dir__, 'inception_v3_fid.ckpt')
    print('Converting...')
    torch_to_mindspore(pt_fp, save=True, save_fp=ckpt_save_fp)
    print('Done! Checkpoint saved in ', ckpt_save_fp)

if __name__=='__main__':
    main()

