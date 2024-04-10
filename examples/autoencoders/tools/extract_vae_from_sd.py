import mindspore as ms

path = "../stable_diffusion_v2/models/sd_v1.5-d0ab7146.ckpt"
ignore_keys = list()
remove_prefix = ["first_stage_model.", "autoencoder."]

sd = ms.load_checkpoint(path)
keys = list(sd.keys())
for k in keys:
    for ik in ignore_keys:
        if k.startswith(ik):
            print("Deleting key {} from state_dict.".format(k))
            del sd[k]

for pname in keys:
    is_vae_param = False
    for pf in remove_prefix:
        if pname.startswith(pf):
            sd[pname.replace(pf, "")] = sd.pop(pname)
            is_vae_param = True
    if not is_vae_param:
        sd.pop(pname)

ms.save_checkpoint(sd, "models/sd1.5_vae.ckpt")
