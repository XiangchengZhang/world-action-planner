import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig
from pathlib import Path


class DummyVideoDataset(Dataset):
    def __init__(self, cfg: DictConfig, split: str = "training") -> None:
        super().__init__()
        self.cfg = cfg
        self.split = split
        self.height = cfg.height
        self.width = cfg.width
        self.n_frames = cfg.n_frames
        self.load_video_latent = cfg.load_video_latent
        self.load_prompt_embed = cfg.load_prompt_embed
        self.image_to_video = cfg.image_to_video
        self.max_text_tokens = cfg.max_text_tokens

    @property
    def metadata_path(self):
        raise ValueError("Dummy dataset does not have a metadata path")

    @property
    def data_root(self):
        raise ValueError("Dummy dataset does not have a data root path")

    def __len__(self) -> int:
        return 10000000  # Return fixed size of 10000000

    def __getitem__(self, idx: int) -> dict:
        # Generate dummy video tensor [T, C, H, W]
        videos = torch.randn(self.n_frames, 3, self.height, self.width)

        # Generate dummy image if needed
        images = videos[:1].clone() if self.image_to_video else None

        output = {
            "prompts": f"A dummy video caption for debugging purpose",
            "videos": videos,
            "video_metadata": {
                "num_frames": self.n_frames,
                "height": self.height,
                "width": self.width,
                "has_caption": True,
            },
            "has_bbox": torch.tensor([False, False]),
            "bbox_render": torch.zeros(2, self.height, self.width),
        }

        if images is not None:
            output["images"] = images

        if self.load_prompt_embed:
            # Generate dummy prompt embeddings [self.max_text_tokens, 4096]
            output["prompt_embeds"] = torch.randn(self.max_text_tokens, 4096)
            output["prompt_embed_len"] = self.max_text_tokens

        if self.load_video_latent:
            # Generate dummy latents
            if self.image_to_video:
                output["image_latents"] = torch.randn(
                    4,
                    self.n_frames // 4,
                    self.height // 8,
                    self.width // 8,
                )
            output["video_latents"] = torch.randn(
                4,
                self.n_frames // 4,
                self.height // 8,
                self.width // 8,
            )

        return output
