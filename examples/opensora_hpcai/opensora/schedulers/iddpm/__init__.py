# Modified from OpenAI's diffusion repos
#     GLIDE: https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py
#     ADM:   https://github.com/openai/guided-diffusion/blob/main/guided_diffusion
#     IDDPM: https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
from typing import Optional

from .diffusion_utils import LossType, ModelMeanType, ModelVarType, get_named_beta_schedule
from .respace import SpacedDiffusion, space_timesteps


def create_diffusion(
    timestep_respacing: Optional[str],
    noise_schedule: str = "linear",
    use_kl: bool = False,
    sigma_small: bool = False,
    predict_xstart: bool = False,
    predict_velocity: bool = False,
    learn_sigma: bool = True,
    rescale_learned_sigmas: bool = False,
    diffusion_steps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    snr_shift_scale: Optional[float] = None,
    rescale_betas_zero_snr: bool = False,
) -> SpacedDiffusion:
    betas = get_named_beta_schedule(noise_schedule, diffusion_steps, beta_start=beta_start, beta_end=beta_end)
    if use_kl:
        loss_type = LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = LossType.RESCALED_MSE
    else:
        loss_type = LossType.MSE
    if timestep_respacing is None or timestep_respacing == "":
        timestep_respacing = [diffusion_steps]

    if predict_xstart:
        model_mean_type = ModelMeanType.START_X
    elif predict_velocity:
        model_mean_type = ModelMeanType.VELOCITY
    else:
        model_mean_type = ModelMeanType.EPSILON

    return SpacedDiffusion(
        use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
        betas=betas,
        model_mean_type=model_mean_type,
        model_var_type=(
            (ModelVarType.FIXED_LARGE if not sigma_small else ModelVarType.FIXED_SMALL)
            if not learn_sigma
            else ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        snr_shift_scale=snr_shift_scale,
        rescale_betas_zero_snr=rescale_betas_zero_snr,
    )
