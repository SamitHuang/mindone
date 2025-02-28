import os
import math
from fractions import Fraction
from typing import Union, Optional, Tuple

import av
import imageio
import numpy as np

__all__ = ["save_videos", "create_video_from_numpy_frames"]


def make_grid(
    tensor,
    nrow = 8,
    padding = 2,
    normalize = False,
    value_range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: float = 0.0,
):
    """
    Make a grid of images.

    Args:
        tensor (np array or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.. typically input range is -1 to 1
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by ``value_range``. Default: ``False``.
        value_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Returns:
        grid (np array): the tensor containing grid of images. if normalize, will be normalized to (0, 1)
    """
    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = np.stack(tensor, axis=0)
    if tensor.ndim == 2:  # single image H x W
        tensor = np.expand_dims(tensor, 0)
    if tensor.ndim == 3:  # single image
        if tensor.shape[0] == 1:  # if single-channel, convert to 3-channel
            tensor = np.concatenate((tensor, tensor, tensor), 0)
        tensor = np.expand_dims(tensor, 0)

    if tensor.ndim == 4 and tensor.shape[1] == 1:  # single-channel images
        tensor = np.concatenate((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.copy()  # avoid modifying tensor in-place
        if value_range is not None and not isinstance(value_range, tuple):
            raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

        def norm_ip(img, low, high):
            img = np.clip(img, a_min=low, a_max=high)
            img = np.divide(np.subtract(img, low), (max(high - low, 1e-5)))
            return img

        def norm_range(t, value_range):
            if value_range is not None:
                t = norm_ip(t, value_range[0], value_range[1])
            else:
                t = norm_ip(t, float(t.min()), float(t.max()))
            return t

        if scale_each is True:
            for i in range(tensor.shape[0]):  # loop over mini-batch dimension
                tensor[i] = norm_range(ttensor[i], value_range)
        else:
            tensor = norm_range(tensor, value_range)

    if tensor.shape[0] == 1:
        return np.squeeze(tensor, 0)

    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    num_channels = tensor.shape[1]
    grid = np.full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value, dtype=np.float32)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            h_start = (y * height + padding)
            h_len = (height - padding)
            w_start = (x * width + padding)
            w_len = (width - padding)
            grid[:, h_start : h_start + h_len, w_start : w_start + w_len] = tensor[k]
            k = k + 1
    return grid



def create_video_from_rgb_numpy_arrays(image_arrays, output_file, fps: Union[int, float] = 30):
    """
    Creates an MP4 video file from a series of RGB NumPy array images.

    Parameters:
    image_arrays (list): A list of RGB NumPy array images.
    output_file (str): The path and filename of the output MP4 video file.
    fps (int): The desired frames per second for the output video. Default is 30.

    Credit to Perlexity
    """
    # Get the dimensions of the first image
    height, width, _ = image_arrays[0].shape

    # Create the output container and video stream
    container = av.open(output_file, mode="w")
    stream = container.add_stream(
        "libx264", rate=Fraction(f"{fps:.4f}")
    )  # BUG: OverflowError: value too large to convert to int
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    # stream.time_base = av.Rational(1, fps)

    # Write the frames to the video stream
    for image in image_arrays:
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    # Flush any remaining frames
    for packet in stream.encode(None):
        container.mux(packet)

    # Close the container
    container.close()


def create_video_from_numpy_frames(frames: np.ndarray, path: str, fps: Union[int, float] = 8, fmt="gif", loop=0):
    """
    Args:
        frames: shape (f h w 3), range [0, 255], order rgb
    """
    if fmt == "gif":
        imageio.mimsave(path, frames, duration=1 / fps, loop=loop)
    elif fmt == "png":
        for i in range(len(frames)):
            imageio.imwrite(path.replace(".png", f"-{i:04}.png"), frames[i])
    elif fmt == "mp4":
        create_video_from_rgb_numpy_arrays(frames, path, fps=fps)


def save_videos(frames: np.ndarray, path: str, fps: Union[int, float] = 8, loop=0, concat=False):
    """
    Save video frames to gif or mp4 files
    Args:
        frames: video frames in shape (b f h w 3), pixel value in [0, 1], RGB mode.
        path:  file path to save the output gif
        fps: frames per sencond in the output gif. 1/fps = display duration per frame
        concat: if True and b>1, all videos will be concatnated in grids and saved as one gif.
        loop: number of loops to play. If 0, it will play endlessly.
    """
    fmt = path.split(".")[-1]
    assert fmt in ["gif", "mp4", "png"]

    # input frames: (b f H W 3), normalized to [0, 1]
    frames = (frames * 255).round().clip(0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if len(frames.shape) == 4:
        create_video_from_numpy_frames(frames, path, fps, fmt, loop)
    else:
        b, f, h, w, _ = frames.shape
        if b > 1:
            if concat:
                canvas = np.array((f, h, w * b, 3), dtype=np.uint8)
                for idx in range(b):
                    canvas[:, :, (w * idx) : (w * (idx + 1)), :] = frames[idx]
                create_video_from_numpy_frames(canvas, path, fps, fmt, loop)
            else:
                for idx in range(b):
                    cur_path = path.replace(f".{fmt}", f"-{idx}.{fmt}")
                    create_video_from_numpy_frames(frames[idx], cur_path, fps, fmt, loop)
        else:
            create_video_from_numpy_frames(frames[0], path, fps, fmt, loop)
