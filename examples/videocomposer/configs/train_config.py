import os

from easydict import EasyDict

cfg = EasyDict(__name__="Config: VideoComposer")

# image is style_image, loca-image is single_image in paper
# cfg.video_compositions = ["text", "mask", "depthmap", "sketch", "motion", "image", "local_image", "single_sketch"]
cfg.video_compositions = ["text", "motion", "image", "local_image"]
cfg.conditions_for_train = ["text", "motion", "image", "local_image"]  # PyTorch + Ascend setting

cfg.midas_checkpoint = "midas_v3_dpt_large-c8fd1049.ckpt"
cfg.pidinet_checkpoint = "table5_pidinet-37904a63.ckpt"
cfg.sketch_simplification_checkpoint = "sketch_simplification_gan-b928fdfa.ckpt"

# dataset
cfg.root_dir = "./demo_video"  # "webvid10m/"
cfg.alpha = 0.7
cfg.misc_size = 384
cfg.depth_std = 20.0
cfg.depth_clamp = 10.0
cfg.hist_sigma = 10.0
cfg.use_image_dataset = False
cfg.alpha_img = 0.7
cfg.resolution = 256
cfg.mean = [0.5, 0.5, 0.5]
cfg.std = [0.5, 0.5, 0.5]

cfg.shuffle = True

# sketch
cfg.sketch_mean = [0.485, 0.456, 0.406]
cfg.sketch_std = [0.229, 0.224, 0.225]

# dataloader
cfg.max_words = 1000
cfg.feature_framerate = 4
cfg.max_frames = 8  # 16
cfg.batch_size = 1
cfg.chunk_size = 64
cfg.num_workers = 8  # not used yet
cfg.prefetch_factor = 2  # not used yet
cfg.seed = 8888

# diffusion
cfg.num_timesteps = 1000
cfg.mean_type = "eps"
cfg.var_type = "fixed_small"  # NOTE: to stabilize training and avoid NaN
cfg.loss_type = "mse"
cfg.clamp = 1.0
cfg.share_noise = False
cfg.use_div_loss = False

cfg.linear_start = 0.00085
cfg.linear_end = 0.0120
cfg.cond_stage_trainable = False
cfg.scale_factor = 0.18215

# classifier-free guidance
cfg.p_zero = 0.9
cfg.guide_scale = 6.0

# stable diffusion
cfg.sd_checkpoint = "sd_v2-1_base-7c8d09ce.ckpt"
cfg.sd_config = {
    "double_z": True,
    "z_channels": 4,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0,
}

# clip vision encoder
cfg.vit_image_size = 224  # for vit-h, it is 224
cfg.vit_patch_size = 14
cfg.vit_dim = 1024
cfg.vit_heads = 16
cfg.vit_layers = 24
cfg.vit_mean = [0.48145466, 0.4578275, 0.40821073]
cfg.vit_std = [0.26862954, 0.26130258, 0.27577711]
cfg.clip_checkpoint = "open_clip_vit_h_14-9bb07a10.ckpt"
cfg.clip_tokenizer = "bpe_simple_vocab_16e6.txt.gz"
cfg.mvs_visual = False

# unet
cfg.unet_in_dim = 4
cfg.unet_concat_dim = 8
cfg.unet_context_dim = 1024
cfg.unet_out_dim = 8 if cfg.var_type.startswith("learned") else 4
cfg.unet_dim = 320
cfg.unet_dim_mult = [1, 2, 4, 4]
cfg.unet_res_blocks = 2
cfg.unet_num_heads = 8
cfg.unet_head_dim = 64
cfg.unet_attn_scales = [1 / 1, 1 / 2, 1 / 4]
cfg.unet_dropout = 0.1
cfg.misc_dropout = (
    0.5  # an independent probability of 0.5 to keep or discard a specific condition.  TODO: two large for finetune?
)
cfg.p_all_zero = 0.1  # we adhere to [26], using a probability of 0.1 to keep all conditions, a probability of 0.1 to discard all conditions
cfg.p_all_keep = 0.1
cfg.temporal_conv = False
cfg.temporal_attn_times = 1
cfg.temporal_attention = True
cfg.use_fps_condition = False
cfg.use_sim_mask = False

# load 2d pretrain
cfg.pretrained = False
cfg.fix_weight = False

# resume
cfg.resume = True
cfg.resume_step = 228000  # 148000
cfg.resume_check_dir = "."
cfg.resume_checkpoint = os.path.join(cfg.resume_check_dir, f"step_{cfg.resume_step}/non_ema_{cfg.resume_step}.pth")
cfg.resume_optimizer = False
if cfg.resume_optimizer:
    cfg.resume_optimizer = os.path.join(cfg.resume_check_dir, f"optimizer_step_{cfg.resume_step}.pt")

# acceleration
cfg.load_from = None
cfg.use_checkpoint = False
cfg.use_sharded_ddp = False
cfg.use_fsdp = False
cfg.use_fp16 = True
cfg.use_adaptive_pool = False  # False (AvgPool2D) is much faster on ms2.0

# training - lr
cfg.learning_rate = 1e-6  # 0.00005 in paper, but we are finetuning.
cfg.scheduler = "cosine_decay"
cfg.end_learning_rate = 1e-7
cfg.warmup_steps = 3
cfg.decay_steps = None  # None for auto compute

# training - optim
cfg.optim = "momentum"  # 'adamw'
cfg.betas = [0.9, 0.98]
cfg.weight_decay = 1e-6  # not mentioned in paper. let's start with small value
cfg.use_ema = False
cfg.ema_decay = 0.9999

cfg.epochs = 50
cfg.ckpt_save_interval = 50  # 1000 # only save last checkpoint for DEBUG
cfg.ouptut_dir = "outputs/train"  # log will be saved here too
cfg.viz_interval = 1000


# logging
cfg.log_interval = 1
composition_strings = "_".join(cfg.video_compositions)
cfg.mode = 0  # 0: Graph mode; 1: Pynative mode
