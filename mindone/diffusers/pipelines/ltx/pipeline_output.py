from dataclasses import dataclass

import mindspore as ms

from mindone.diffusers.utils import BaseOutput


@dataclass
class LTXPipelineOutput(BaseOutput):
    r"""
    Output class for LTX pipelines.

    Args:
        frames (`ms.Tensor`, `np.ndarray`, or List[List[PIL.Image.Image]]):
            List of video outputs - It can be a nested list of length `batch_size,` with each sub-list containing
            denoised PIL image sequences of length `num_frames.` It can also be a NumPy array or MindSpore tensor of
            shape `(batch_size, num_frames, channels, height, width)`.
    """

    frames: ms.Tensor
