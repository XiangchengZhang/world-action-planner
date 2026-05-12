from pathlib import Path
from typing import Any, Dict, List, Tuple
import random
import threading
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
from einops import rearrange
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms


# Must import after torch because this can sometimes lead to a nasty segmentation fault, or stack smashing error
# Very few bug reports but it happens. Look in decord Github issues for more relevant information.
import decord  # isort:skip

decord.bridge.set_bridge("torch")

from .video_base import VideoDataset


class CondVideoDataset(VideoDataset):
    def __init__(self, cfg: DictConfig, split: str = "training") -> None:
        super().__init__(cfg, split)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        return self.load_record(record)

    def load_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        # Load video data - either raw or preprocessed latents
        videos, conds = self._load_video_cond(record)
        # images = videos[:1].clone() if self.image_to_video else None
        image_latents, video_latents = None, None
        video_metadata = {
            "num_frames": videos.shape[0],
            "height": videos.shape[2],
            "width": videos.shape[3],
        }

        # Load prompt data - either raw or preprocessed embeddings
        prompts = self.id_token
        prompt_embeds = None
        prompt_embed_len = None
        negative_prompt_embeds = None
        negative_prompt_embed_len = None
        if self.load_prompt_embed:
            prompt_embeds, prompt_embed_len = self._load_prompt_embed(record)
            negative_prompt_embeds, negative_prompt_embed_len = self._load_prompt_embed(record, negative=True)


        has_bbox, bbox_render = self._render_bbox(record)

        output = {
            "videos": videos,
            "conds": conds,
            "video_metadata": video_metadata,
            "bbox_render": bbox_render,
            "has_bbox": has_bbox,
            "video_path": str(self.data_root / record["video_path"]),
        }

        if prompts is not None:
            output["prompts"] = prompts
        # if images is not None:
        #     output["images"] = images
        if prompt_embeds is not None:
            output["prompt_embeds"] = prompt_embeds
            output["prompt_embed_len"] = prompt_embed_len
        if negative_prompt_embeds is not None:
            output["negative_prompt_embeds"] = negative_prompt_embeds
            output["negative_prompt_embed_len"] = negative_prompt_embed_len
        if image_latents is not None:
            output["image_latents"] = image_latents
        if video_latents is not None:
            output["video_latents"] = video_latents

        return output
    
    def pad_actions(self, actions: torch.Tensor):
        import torch.nn.functional as F
        action_dim = self.cfg.get('action_dim', 7)
        if actions.shape[-1] > action_dim:
            raise ValueError(f"Action dimension {actions.shape[-1]} is larger than expected {action_dim}.")
        elif actions.shape[-1] < action_dim:
            if actions.shape[-1] == 7:
                actions = F.pad(actions, (0, action_dim - actions.shape[-1]), "constant", 0)
            elif actions.shape[-1] == 14 or actions.shape[-1] == 24:
                half_actions = actions.chunk(2, dim=-1)
                half_actions = [F.pad(half_action, (0, action_dim // 2 - half_action.shape[-1]), "constant", 0) for half_action in half_actions]
                actions = torch.cat(half_actions, dim=-1)
            else:
                raise NotImplementedError(f"Padding for action dimension {actions.shape[-1]} is not implemented.")
        # print("[DEBUG] [Action shape]", actions.shape)
        return actions
                



    def _load_video_cond(self, record: Dict[str, Any]) -> torch.Tensor:
        """
        Given a record, return a tensor of shape (n_frames, 3, H, W)
        """

        video_path = self.data_root / record["video_path"]
        cond_path = self.data_root / record['cond_path']
        video_reader = decord.VideoReader(uri=video_path.as_posix())
        cond_reader = decord.VideoReader(uri=cond_path.as_posix())

        n_frames = len(video_reader)
        assert n_frames == len(cond_reader), "Video and cond length mismatch"
        start = record.get("trim_start", 0)
        end = record.get("trim_end", n_frames)
        indices = self._temporal_sample(end - start, record["fps"])
        indices = list(start + indices)
        frames = video_reader.get_batch(indices)
        cond_frames = cond_reader.get_batch(indices)

        # do some padding
        if len(frames) != self.n_frames:
            raise ValueError(
                f"Expected {len(frames)=} to be equal to {self.n_frames=}."
            )

        # crop if specified in the record
        if "crop_top" in record and "crop_bottom" in record:
            frames = frames[:, record["crop_top"] : record["crop_bottom"]]
            cond_frames = cond_frames[:, record["crop_top"] : record["crop_bottom"]]
        if "crop_left" in record and "crop_right" in record:
            frames = frames[:, :, record["crop_left"] : record["crop_right"]]
            cond_frames = cond_frames[:, :, record["crop_left"] : record["crop_right"]]

        frames = frames.float().permute(0, 3, 1, 2).contiguous() / 255.0
        cond_frames = cond_frames.float().permute(0, 3, 1, 2).contiguous() / 255.0
        if "has_bbox" in record and record["has_bbox"]:
            frames = self.no_augment_transforms(frames)
            cond_frames = self.no_augment_transforms(cond_frames)
        else:
            frames = self.augment_transforms(frames)
            cond_frames = self.augment_transforms(cond_frames)
        frames = self.img_normalize(frames)
        cond_frames = self.img_normalize(cond_frames)

        return frames, cond_frames


    def download(self):
        """
        Automatically download the dataset to self.data_root. Optional.
        """
        raise NotImplementedError(
            "Automatic download not implemented for this dataset."
        )
