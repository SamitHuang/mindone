import csv
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from mindspore.dataset.vision import CenterCrop, Inter, Normalize, Resize

# FIXME: remove in future when mindone is ready for install
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))
from mindone.data import BaseDataset
from mindone.data.video_reader import VideoReader

_logger = logging.getLogger(__name__)


class ImageVideo2VideoDataset(BaseDataset):
    def __init__(
        self,
        csv_path: str,
        video_folder: str,
        text_emb_folder: Optional[str] = None,
        sample_n_frames: int = 16,
        sample_stride: int = 4,
        frames_mask_generator: Optional[Callable[[int], np.ndarray]] = None,
        *,
        output_columns: List[str],
    ):
        self._data = self._read_data(video_folder, csv_path)
        self._frames = sample_n_frames
        self._stride = sample_stride
        self._min_length = (self._frames - 1) * self._stride + 1
        self._filter_videos()
        self._text_emb_folder = text_emb_folder
        self._fmask_gen = frames_mask_generator

        self.output_columns = output_columns

    @staticmethod
    def _read_data(data_dir: str, csv_path: str) -> List[dict]:
        with open(csv_path, "r") as csv_file:
            try:
                data = [
                    {**item, "video": os.path.join(data_dir, item["video"]), "length": int(item["length"])}
                    for item in csv.DictReader(csv_file)
                ]
            except KeyError as e:
                _logger.error("CSV file must have a column `video` with paths to the videos")
                raise e

        return data

    def _filter_videos(self):
        old_len = len(self._data)
        self._data = [item for item in self._data if item["length"] >= self._min_length]
        if len(self._data) < old_len:
            _logger.info(
                f"Filtered out {old_len - len(self._data)} videos as they don't match the minimum length"
                f" requirement: {self._min_length} frames ((num frames - 1) x stride + 1)"
            )

    def __getitem__(self, idx: int) -> Tuple[Any, ...]:
        data = self._data[idx].copy()
        if self._text_emb_folder:
            with np.load(Path(data["video"]).stem + ".npz") as td:
                data.update({"caption": td["text_emb"], "mask": td["mask"]})

        with VideoReader(data["video"]) as reader:
            start_pos = random.randint(0, len(reader) - self._min_length)
            data["video"] = reader.fetch_frames(num=self._frames, start_pos=start_pos, step=self._stride)
            data.update(
                {"fps": np.array(reader.fps, dtype=np.float32), "num_frames": np.array(self._frames, dtype=np.float32)}
            )

        if self._fmask_gen is not None:
            data["frames_mask"] = self._fmask_gen(self._frames)

        return tuple(data[c] for c in self.output_columns)

    def __len__(self):
        return len(self._data)

    def train_transforms(
        self, target_size: Tuple[int, int], tokenizer: Optional[Callable[[str], np.ndarray]] = None
    ) -> List[dict]:
        transforms = [
            {
                "operations": [
                    Resize(min(target_size), interpolation=Inter.BILINEAR),
                    CenterCrop(target_size),
                    lambda x: (x / 255.0).astype(np.float32),  # ms.ToTensor() doesn't support 4D data
                    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    lambda x: np.transpose(x, (0, 3, 1, 2)),  # ms.HWC2CHW() doesn't support 4D data
                ],
                "input_columns": ["video"],
            },
            {
                "operations": [
                    lambda video: (
                        video,
                        np.array(video.shape[-2], dtype=np.float32),
                        np.array(video.shape[-1], dtype=np.float32),
                        np.array(video.shape[-1] / video.shape[-2], dtype=np.float32),
                    )
                ],
                "input_columns": ["video"],
                "output_columns": ["video", "height", "width", "ar"],
            },
        ]

        if "caption" in self.output_columns and not self._text_emb_folder:
            if tokenizer is None:
                raise RuntimeError("Please provide a tokenizer for text data in `train_transforms()`.")
            transforms.append({"operations": [tokenizer], "input_columns": ["caption"]})

        return transforms
