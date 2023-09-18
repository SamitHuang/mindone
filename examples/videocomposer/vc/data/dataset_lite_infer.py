import logging
import os
import random
from typing import List

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from ..annotator.motion import extract_motion_vectors
from .transforms import create_transforms
from .utils import get_video_paths_captions


_logger = logging.getLogger(__name__)


class VideoDatasetLiteInfer(object):
    def __init__(
        self,
        video_paths: List=None,   
        captions: List=None, 
        #neg_prompts: List=None,
        root_dir=None,
        max_words=30,
        feature_framerate=1,
        max_frames=16,
        image_resolution=224,
        transforms=None,
        mv_transforms=None,
        misc_transforms=None,
        vit_transforms=None,
        vit_image_size=336,
        misc_size=384,
        mvs_visual=False,
        tokenizer=None,
    ):
        """
        Args:
            root_dir: dir containing csv file which records video path and caption.
        """

        self.tokenizer = tokenizer
        self.max_words = max_words
        self.feature_framerate = feature_framerate
        self.max_frames = max_frames
        self.image_resolution = image_resolution
        self.transforms = transforms
        self.mv_transforms = mv_transforms
        self.misc_transforms = misc_transforms
        self.vit_transforms = vit_transforms
        self.vit_image_size = vit_image_size
        self.misc_size = misc_size
        self.mvs_visual = mvs_visual

        if root_dir is not None and os.path.exists(root_dir):
            video_paths, captions = get_video_paths_captions(root_dir)
        else:
            video_paths, captions = video_paths, captions 

        num_samples = len(video_paths)
        self.video_cap_pairs = [[video_paths[i], captions[i]] for i in range(num_samples)]
        self.tokenizer = tokenizer  # bpe

    def tokenize(self, text):
        tokens = self.tokenizer(text, padding="max_length", max_length=77)["input_ids"]

        return tokens

    def __len__(self):
        return len(self.video_cap_pairs)

    def __getitem__(self, index):
        video_key, cap_txt = self.video_cap_pairs[index]

        feature_framerate = self.feature_framerate
        if os.path.exists(video_key):
            vit_image, video_data, misc_data, mv_data = self._get_video_train_data(
                video_key, feature_framerate, self.mvs_visual
            )
        else:  # use dummy data
            _logger.warning(f"The video path: {video_key} does not exist or no video dir provided!")
            vit_image = np.zeros((3, self.vit_image_size, self.vit_image_size), dtype=np.float32)  # noqa
            video_data = np.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution), dtype=np.float32)
            misc_data = np.zeros((self.max_frames, 3, self.misc_size, self.misc_size), dtype=np.float32)
            mv_data = np.zeros((self.max_frames, 2, self.image_resolution, self.image_resolution), dtype=np.float32)

        # adapt for training, output element must map the order of model construct input
        caption_tokens = self.tokenize(cap_txt)

        return video_data, caption_tokens, feature_framerate, mv_data, misc_data

    def _get_video_train_data(self, video_key, feature_framerate, viz_mv):
        filename = video_key
        frame_types, frames, mvs, mvs_visual = extract_motion_vectors(
            input_video=filename, fps=feature_framerate, viz=viz_mv
        )

        total_frames = len(frame_types)
        start_indices = np.where(
            (np.array(frame_types) == "I") & (total_frames - np.arange(total_frames) >= self.max_frames)
        )[0]
        start_index = np.random.choice(start_indices)
        indices = np.arange(start_index, start_index + self.max_frames)

        # note frames are in BGR mode, need to trans to RGB mode
        # TODO: we don't need these input video frames in inference. Removing it can speed up a little.
        frames = [Image.fromarray(frames[i][:, :, ::-1]) for i in indices]
        mvs = [mvs[i].astype(np.float32) for i in indices]  # h, w, 2

        have_frames = len(frames) > 0
        middle_index = int(len(frames) / 2)
        if have_frames:
            ref_frame = frames[middle_index]
            vit_image = self.vit_transforms(ref_frame)[0]
            misc_imgs = np.stack([self.misc_transforms(frame)[0] for frame in frames], axis=0)
            frames = np.stack([self.transforms(frame)[0] for frame in frames], axis=0)
            mvs = np.stack([self.mv_transforms(mv).transpose((2, 0, 1)) for mv in mvs], axis=0)
        else:
            raise RuntimeError(f"Got no frames from {filename}!")

        video_data = np.zeros((self.max_frames, 3, self.image_resolution, self.image_resolution), dtype=np.float32)
        mv_data = np.zeros((self.max_frames, 2, self.image_resolution, self.image_resolution), dtype=np.float32)
        misc_data = np.zeros((self.max_frames, 3, self.misc_size, self.misc_size), dtype=np.float32)
        if have_frames:
            video_data[: len(frames), ...] = frames
            misc_data[: len(frames), ...] = misc_imgs
            mv_data[: len(frames), ...] = mvs

        return vit_image, video_data, misc_data, mv_data

def read_and_transform_image(path, transform=None):
    if path is None:
        return None

    img = Image.open(open(path, mode="rb")).convert("RGB")
    if transform is not None:
        img = transform(img)

    return img

def load_data(
        cfg, 
        tokenizer, 
        video_path=None, 
        prompt=None, 
        neg_prompt=None, 
        single_image_path=None,
        style_image_path=None,
        expand_batch_axis=True,
        ):
    ''' 
    Returns: numpy data for a single data sample, without batch sampling
    '''
    infer_transforms, misc_transforms, mv_transforms, vit_transforms = create_transforms(cfg)
    ds = VideoDatasetLiteInfer(
        video_paths=[video_path],
        captions=[prompt],
        max_words=cfg.max_words,
        feature_framerate=cfg.feature_framerate,
        max_frames=cfg.max_frames,
        image_resolution=cfg.resolution,
        transforms=infer_transforms,
        mv_transforms=mv_transforms,
        misc_transforms=misc_transforms,
        vit_transforms=vit_transforms,
        vit_image_size=cfg.vit_image_size,
        misc_size=cfg.misc_size,
        mvs_visual=cfg.mvs_visual,
        tokenizer=tokenizer,
    )
    
    # TODO: add BatchSampler an DataLoader (for numpy)
    assert len(ds) == 1, 'Currently only support single-sample inference'

    video_frames, prompt_data, fps, motion_vectors, misc_data = ds.__getitem__(0)
    negative_prompt_data = ds.tokenize(neg_prompt)

    # by vision.ToTensor(), the transformed image is in nchw format and np.float32 type by default
    single_image = read_and_transform_image(single_image_path, misc_transforms)[0]
    style_image = read_and_transform_image(style_image_path, vit_transforms)[0] # TODO: it uses CenterCrop, need to limit input size > 224x224
    
    # the data type must be the same as the one defined in mindir export
    data = {"video_frames": video_frames,
            "prompt_data": np.array(prompt_data, np.int32),
            "negative_prompt_data": np.array(negative_prompt_data, np.int32),
            "fps": fps,
            "motion_vectors": motion_vectors,
            "single_image": single_image,
            "style_image": style_image,
            }
    
    if expand_batch_axis:
        for k in data:
            if isinstance(data[k], np.ndarray) and k != 'fps':
                data[k] = np.expand_dims(data[k], axis=0)         
                #print("D--: ", k, data[k].shape)
            else:
                print(f"Got non-numpy data {k}, type: ", type(data[k]))

    data["fps"] = np.array(data["fps"], np.int32) 
    
    return data
