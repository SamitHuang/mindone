import ms
import torch
import os

# convert based on order correspondance
def convert_pt_ms_state_dict(pt_ckpt_path, pt_pinfo_path, ms_pinfo_path, add_ldm_prefix=False, ldm_prefix='model.diffusion_model.'):
    pt_sd = torch.load(pt_ckpt_path)

    # some name mapping rules, which are found from param name printing observation (case by case)
    def _param_info_parser(param_info_txt):
        param_info = []
        with open(param_info_txt) as fp:
            for line in fp:
                pname, pshape, ptype = line.strip().split('#')
                param_info.append((pname, pshape, ptype))
        return param_info

    pt_param_info = _param_info_parser(pt_pinfo_path)
    ms_param_info = _param_info_parser(ms_pinfo_path)

    print('Converting ', pt_ckpt_path)
    target_data = []
    for i, (pt_pname, pt_pshape, pt_ptype) in enumerate(pt_param_info):
        ms_pname, ms_pshape, ms_dtype = ms_param_info[i]
        assert pt_pshape==ms_pshape, f"pt and ms param shape mismatch, please check order correspondance. pt: {pt_pname}, {pt_pshape}, ms: {ms_pname}, {ms_pshape}. "
        assert pt_pname in pt_sd, f"param not in torch ckpt, please check: {pt_pname}"
        if add_ldm_prefix:
            ms_pname = ldm_prefix + ms_pname
        target_data.append({"name": ms_pname, "data": ms.Tensor(pt_sd[pt_pname].detach().numpy())})

    ms_path = pt_ckpt_path.replace(".ckpt", "_ms.ckpt")
    ms.save_checkpoint(target_data, ms_path)

    print('ms ckpt saved in', ms_path)
    return ms_path

src_pt_ckpt = '../models/Motion_Module/animatediff/mm_sd_v15_v2.ckpt'
torch_names_txt = './torch_mm_params.txt'
ms_names_txt = './ms_mm_params.txt'

ms_path = convert_pt_ms_state_dict(src_pt_ckpt, torch_names_txt, ms_names_txt)
