'''
AnimateDiff inference pipeline
'''
import logging
import time
import argparse
import datetime
import inspect
import os
from omegaconf import OmegaConf
from tqdm.auto import tqdm
import csv, pdb, glob
import math
from pathlib import Path
import numpy as np

from ldm.modules.logger import set_logger
from ldm.modules.train.tools import set_random_seed
from ldm.util import instantiate_from_config, str2bool
from ldm.pipelines.load_models import load_model_from_config
from ldm.pipelines.infer_engine import SDText2Img
from ldm.pipelines.image_utils import VaeImageProcessor

import mindspore as ms

#import torch
#import diffusers
#from diffusers import AutoencoderKL, DDIMScheduler
#from transformers import CLIPTextModel, CLIPTokenizer

#from animatediff.models.unet import UNet3DConditionModel
#from animatediff.pipelines.pipeline_animation import AnimationPipeline
#from animatediff.utils.util import save_videos_grid
#from animatediff.utils.util import load_weights
#from diffusers.utils.import_utils import is_xformers_available
#from einops import rearrange, repeat

logger = logging.getLogger(__name__)

def init_env(args):
    # no parallel mode currently
    ms.set_context(mode=args.ms_mode)  # needed for MS2.0
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.set_context(
        mode=args.ms_mode,
        device_target=args.target_device,
        device_id=device_id,
        # max_device_memory="30GB", # adapt for 910b
    )
    if args.target_device == 'Ascend':
        ms.set_context(ascend_config={"precision_mode": "allow_fp32_to_fp16"})  # Only effective on Ascend 901B

    return device_id


def main(args):
    # set work dir and save dir 
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir = f"samples/{Path(args.config).stem}-{time_str}"

    # 0. parse and merge config
    # 1) sd config, 2) db ckpt path, 3) lora ckpt path, 4) mm ckpt path, 5) unet additional args, 6) noise schedule args
    config = OmegaConf.load(args.config)
    task_name = list(config.keys())[0]
    ad_config = config[task_name]
    # print("D--: ", ad_config)

    dreambooth_path = ad_config.get("dreambooth_path", "")
    lora_model_path = ad_config.get("lora_model_path", "")
    lora_alpha = lora_scale = ad_config.get("lora_alpha", 0.8)

    motion_module_paths = ad_config.get("motion_module", "")

    seeds, steps, guidance_scale = ad_config.get("seed", 0), ad_config.steps, ad_config.guidance_scale 
    prompts = ad_config.prompt
    n_prompts = ad_config.n_prompt
    seeds = [seeds] * len(prompts) if isinstance(seeds, int) else seeds

    sd_config = OmegaConf.load(args.sd_config) 
    sd_model_path = args.pretrained_model_path

    if dreambooth_path != "":
        if os.path.exists(dreambooth_path):
            sd_model_path = dreambooth_path  # DB params naming rule is the same sd ldm  
        else:
            logger.warning(f"dreambooth path {dreambooth_path} not exist.")
    use_lora = True if lora_model_path != "" else False
    #print("D--: ", lora_model_path, use_lora)

    inference_config = OmegaConf.load(ad_config.get("inference_config", args.inference_config))
    unet_additional_kwargs = inference_config.unet_additional_kwargs
    noise_scheduler_kwargs = inference_config.noise_scheduler_kwargs
    
    # TODO: merge unet addition kwargs to sd_confg

    # 1. init env 
    init_env(args)
    set_random_seed(42)

    # 2. build model components for ldm  
    # 1)  vae, text encoder, and unet
    # TODO: change mixed precision. fp32 at first?
    sd_model = load_model_from_config(
        sd_config,
        ckpt=sd_model_path,
        use_lora=use_lora,
        lora_only_ckpt=lora_model_path,
        lora_scale=lora_scale,
    )
    text_encoder = sd_model.cond_stage_model
    unet = sd_model.model
    vae = sd_model.first_stage_model
    img_processor = VaeImageProcessor()

    if args.target_device!= "Ascend":
        unet.to_float(ms.float32)
        vae.to_float(ms.float32)

    # 2) ddim sampler
    sampler_config = OmegaConf.load("configs/inference/scheduler/ddim.yaml")
    sampler_config.beta_start = noise_scheduler_kwargs.beta_start
    sampler_config.beta_end = noise_scheduler_kwargs.beta_end
    sampler_config.beta_schedule = noise_scheduler_kwargs.beta_schedule

    scheduler = instantiate_from_config(sampler_config)
    timesteps = scheduler.set_timesteps(steps)

    # 3. build inference pipeline 
    pipeline = SDText2Img(
        text_encoder,
        unet,
        vae,
        scheduler,
        scale_factor=sd_model.scale_factor,
        num_inference_steps=steps,
    )

    # 4. run sampling for multiple samples
    num_prompts = len(prompts)
    bs = 1 # batch size
    sample_idx = 0
    for i in range(num_prompts):
        ms.set_seed(seeds[i])
        prompt = prompts[i] 
        n_prompt = n_prompts[i]
        
        # creat inputs
        inputs = {}
        inputs["prompt"] = prompt
        inputs["prompt_data"] = sd_model.tokenize([prompt] * bs)
        inputs["negative_prompt"] = n_prompt
        inputs["negative_prompt_data"] = sd_model.tokenize([n_prompt] * bs)
        inputs["timesteps"] = timesteps
        inputs["scale"] = ms.Tensor(guidance_scale, ms.float16)

        noise = np.random.randn(bs, 4, args.H // 8, args.W // 8)
        inputs["noise"] = ms.Tensor(noise, ms.float16)

        logger.info("Sampling prompt: ", prompts[i])
        start_time = time.time()
        
        # infer
        x_samples = pipeline(inputs)
        x_samples = img_processor.postprocess(x_samples) # transpose, convert to PIL object
        sample = x_samples[0]

        end_time = time.time()
        
        # save result
        os.makedirs(save_dir, exist_ok=True)
        prompt = "-".join((prompt.replace("/", "").split(" ")[:10]))
        save_fp = f"{save_dir}/{sample_idx}-{prompt}.png"
        sample.save(save_fp)
        #save_videos_grid(sample, f"{save_dir}/sample/{sample_idx}-{prompt}.gif")
        logger.info(f"save to {save_fp}")
        sample_idx += 1

        logger.info(
            "Time cost: {:.3f}s".format(end_time - start_time)
        )

    logger.info(f"Done! All generated images are saved in: {save_dir}" f"\nEnjoy.")
    OmegaConf.save(config, f"{save_dir}/config.yaml")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="models/stable_diffusion/sd_v1.5-d0ab7146.ckpt",)
    parser.add_argument("--inference_config",      type=str, default="configs/inference/inference-v1.yaml")    
    parser.add_argument("--config",                type=str, default="configs/prompts/1-ToonYou.yaml")
    
    parser.add_argument("--L", type=int, default=16 )
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--H", type=int, default=512)

    # Use ldm config method instead of diffusers and transformers 
    parser.add_argument("--sd_config", type=str, default="configs/stable_diffusion/v1-inference.yaml")    

    # MS new args
    parser.add_argument("--target_device", type=str, default="GPU", help="Ascend or GPU")
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )

    args = parser.parse_args()
    
    print(args)

    main(args)



