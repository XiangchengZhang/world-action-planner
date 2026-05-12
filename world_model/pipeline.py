import yaml
from pathlib import Path
from typing import List, NamedTuple, Union

import PIL
import numpy as np
import torch
from omegaconf import OmegaConf

from experiments.exp_video import compatible_algorithms, compatible_datasets
from utils.ckpt_utils import is_run_id, retrive_checkpoint

PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,  # H W C [0, 1] or [0, 255]
    torch.Tensor,  # C H W [0, 1]
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
]

PipelineVideoOutput = NamedTuple(
    "PipelineVideoOutput",
    full_video=List[PipelineImageInput],  # (T, C, H, W), range [0, 255]
    pred_frames=List[PipelineImageInput],
    pred_panels=List[List[PipelineImageInput]],
)


def to_pt(frame: PipelineImageInput, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert image input to a torch tensor of shape (C, H, W), range [0, 1]."""
    if isinstance(frame, PIL.Image.Image):
        frame = torch.tensor(
            np.array(frame).astype(np.float32) / 255.0,
            device=device,
            dtype=dtype,
        ).permute(2, 0, 1)
    elif isinstance(frame, np.ndarray):
        if frame.dtype == np.uint8:
            frame = frame.astype(np.float32) / 255.0
        frame = torch.tensor(frame, device=device, dtype=dtype).permute(2, 0, 1)
    elif isinstance(frame, torch.Tensor):
        frame = frame.to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported frame type: {type(frame)}")
    return frame


def to_video_tensor(
    video_input: List[PipelineImageInput],
    device: torch.device,
    dtype: torch.dtype,
    name: str = "video_input",
) -> torch.Tensor:
    """Convert image list input to a video tensor of shape (T, C, H, W)."""
    if not isinstance(video_input, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of images.")
    if not video_input:
        raise ValueError(f"{name} must contain at least one image.")

    video_frames = []
    for frame in video_input:
        frame = to_pt(frame, device, dtype)  # (C, H, W)
        video_frames.append(frame)
    video_tensor = torch.stack(video_frames, dim=0)  # (T, C, H, W)
    return video_tensor


class VideoPredictionPipeline:
    def __init__(
        self,
        cfg_path,
        checkpoint_path,
        device="cuda",
        dtype=torch.bfloat16,
        overrides=None,
    ):
        self.device = device
        self.dtype = dtype
        self.cfg_path = Path(cfg_path)
        self.root_cfg = OmegaConf.load(cfg_path)

        # update cfg with overrides list (e.g. ["algorithm.hist_guidance=0.5", "algorithm.scale=1"])
        for override in (overrides or []):
            k, v = override.split("=", 1)
            OmegaConf.update(self.root_cfg, k, yaml.safe_load(v), merge=True)

        OmegaConf.resolve(self.root_cfg)
        print(OmegaConf.to_yaml(self.root_cfg))

        # add checkpoint path
        self.root_cfg.algorithm.model.tuned_ckpt_path = str(Path(checkpoint_path).resolve())

        hydra_runtime_cfg = OmegaConf.load(str(cfg_path).replace("config", "hydra"))
        self.algorithm_name = hydra_runtime_cfg.hydra.runtime.choices.algorithm
        self.algorithm = compatible_algorithms[self.algorithm_name](self.root_cfg.algorithm)
        self.algorithm.configure_model()
        self.algorithm = self.algorithm.to(device=self.device, dtype=self.dtype)
        print(f"Loaded algorithm: {self.algorithm_name} to device {self.device}.")

        self.dataset_name = hydra_runtime_cfg.hydra.runtime.choices.dataset
        self.dataset = compatible_datasets[self.dataset_name](self.root_cfg.dataset, split="all")
        print(self)

    def __repr__(self):
        return f"""
        #############Pipeline Config################
        Width: {self.dataset.cfg.width}, Height: {self.dataset.cfg.height},
        FPS: {self.dataset.fps}, override_fps: {self.dataset.cfg.download.get("override_fps", "None")},
        Algorithm: {self.algorithm_name}, cond_mode: {self.algorithm.diffusion_forcing.cond_mode},
        context_len: {self.algorithm.context_len}, n_frames: {self.algorithm.n_frames}, max_frames: {self.algorithm.max_frames}, hist_len: {self.algorithm.hist_len}
        hist_guidance: {self.algorithm.hist_guidance}
        ############################################
        """

    @staticmethod
    def from_pretrained(load_id, entity=None, project=None, overrides=None, **kwargs):
        if is_run_id(load_id):
            run_path = f"{entity}/{project}/{load_id}"
            checkpoint_path = retrive_checkpoint(run_path, "outputs/checkpoint_links", "latest")
        elif Path(load_id).exists():
            checkpoint_path = Path(load_id).resolve()
        else:
            raise ValueError(
                f"Checkpoint '{load_id}' does not exist. Download checkpoints first, or pass a valid local "
                "checkpoint path such as data/ckpts/libero_90_base/checkpoints/latest.ckpt."
            )

        if checkpoint_path is None or not checkpoint_path.exists():
            raise ValueError(f"Checkpoint path {checkpoint_path} does not exist.")

        cfg_path = checkpoint_path.parent.parent / ".hydra" / "config.yaml"
        if not cfg_path.exists():
            raise ValueError(f"Expected Hydra config next to checkpoint at {cfg_path}.")

        return VideoPredictionPipeline(
            cfg_path=str(cfg_path),
            checkpoint_path=str(checkpoint_path),
            overrides=overrides,
            **kwargs,
        )

    def __call__(
        self,
        history_frames: List[PipelineImageInput],
        history_conds: List[PipelineImageInput],
        future_conds: List[PipelineImageInput],
        return_type="pil",
    ) -> PipelineVideoOutput:
        """
        Generate future video frames given history frames and conditions.
        :param history_frames: list of history images
        :param history_conds: list of history condition images
        :param future_conds: list of future condition images
        """
        videos, conds = self.prepare_videos(history_frames, history_conds, future_conds)
        batch = self.build_batch(videos, conds)
        pred_videos = (
            self.algorithm.predict_seq(batch).float().squeeze(0).clamp(-1, 1) * 0.5 + 0.5
        )
        return self.postprocess(pred_videos, return_type=return_type)

    def postprocess(self, pred_videos, return_type="pil"):
        from utils.video_utils import pad_video
        pred_frames = pred_videos[self.algorithm.context_len:]  # (T_pred, C, H, W)
        pred_frames = (pred_frames.cpu().detach().numpy() * 255.0).astype(np.uint8)  # (T, C, H, W)
        pred_frame_list = [pred_frames[i].transpose(1, 2, 0) for i in range(pred_frames.shape[0])]
        pred_panels_list = [
            [
                frame[: frame.shape[0] // 2, : frame.shape[1] // 2],
                frame[: frame.shape[0] // 2, frame.shape[1] // 2 :],
                frame[frame.shape[0] // 2 :, : frame.shape[1] // 2],
                frame[frame.shape[0] // 2 :, frame.shape[1] // 2 :],
            ]
            for frame in pred_frame_list
        ]
        if return_type == "pil":
            pred_frame_list = [PIL.Image.fromarray(frame) for frame in pred_frame_list]
            pred_panels_list = [
                [PIL.Image.fromarray(panel) for panel in panels]
                for panels in pred_panels_list
            ]
        elif return_type == "np":
            pass
        else:
            raise ValueError(f"Unsupported return_type: {return_type}")
        pred_videos = pad_video(pred_videos.unsqueeze(0), pad_len=self.algorithm.context_len).squeeze(0)
        pred_videos = (pred_videos.cpu().detach().numpy().transpose(0, 2, 3, 1) * 255.0).astype(np.uint8)
        pred_videos = list(pred_videos)  # unbind first dimension only, keep arrays
        return PipelineVideoOutput(
            full_video=pred_videos,
            pred_frames=pred_frame_list,
            pred_panels=pred_panels_list,
        )

    def build_batch(self, videos, conds, prompt=None):
        from torch.utils.data.dataloader import default_collate
        output = {
            "videos": videos,
            "conds": conds,
        }
        prompt_embeds, prompt_embed_len = self.dataset._load_prompt_embed({})
        negative_prompt_embeds, negative_prompt_embed_len = self.dataset._load_prompt_embed(
            {},
            negative=True,
        )
        if prompt is not None:
            output["prompts"] = prompt
        if prompt_embeds is not None:
            output["prompt_embeds"] = prompt_embeds
            output["prompt_embed_len"] = prompt_embed_len
        if negative_prompt_embeds is not None:
            output["negative_prompt_embeds"] = negative_prompt_embeds
            output["negative_prompt_embed_len"] = negative_prompt_embed_len
        batch = default_collate([output])
        batch = {
            k: v.to(
                device=self.device,
                dtype=self.dtype if not v.dtype == torch.long else torch.long,
            )
            if isinstance(v, torch.Tensor)
            else v
            for k, v in batch.items()
        }
        return batch

    def prepare_videos(self, history_frames, history_conds, future_conds):
        history_frames = to_video_tensor(history_frames, self.device, self.dtype, "history_frames")
        history_conds = to_video_tensor(history_conds, self.device, self.dtype, "history_conds")
        future_conds = to_video_tensor(future_conds, self.device, self.dtype, "future_conds")
        if history_frames.shape[0] != history_conds.shape[0]:
            raise ValueError(
                "history_frames and history_conds must have the same length. "
                f"Got {history_frames.shape[0]} and {history_conds.shape[0]}."
            )

        source_fps = self.dataset.cfg.download.get("override_fps", 20)
        model_fps = self.dataset.fps
        n_context_frames = round(self.algorithm.context_len / model_fps * source_fps)
        if history_frames.shape[0] < n_context_frames:
            raise ValueError(
                "Not enough history frames for this checkpoint. "
                f"Need at least {n_context_frames} source frames at {source_fps} FPS "
                f"to build {self.algorithm.context_len} model context frames; "
                f"got {history_frames.shape[0]}."
            )

        context_indices = torch.linspace(
            -n_context_frames,
            -1,
            steps=self.algorithm.context_len,
            device=self.device,
        ).long()
        history_frames = history_frames[context_indices]
        history_conds = history_conds[context_indices]

        pred_model_frames = round(future_conds.shape[0] / source_fps * model_fps)
        if pred_model_frames <= 0:
            raise ValueError("future_conds must contain enough frames for at least one model timestep.")
        if pred_model_frames % 4 != 0:
            raise ValueError(f"pred_model_frames must be a multiple of 4. Got {pred_model_frames}.")
        if hasattr(self.algorithm, "pred_len") and pred_model_frames < self.algorithm.pred_len:
            raise ValueError(
                "Not enough future conditions for this checkpoint. "
                f"Need at least {self.algorithm.pred_len} model frames; got {pred_model_frames}."
            )

        pred_future_indices = torch.linspace(
            0,
            future_conds.shape[0] - 1,
            steps=pred_model_frames,
            device=self.device,
        ).long()
        future_conds = future_conds[pred_future_indices]
        future_frames = torch.zeros_like(future_conds)
        videos = torch.cat([history_frames, future_frames], dim=0)  # (B, T, C, H, W)
        conds = torch.cat([history_conds, future_conds], dim=0)
        videos = self.dataset.augment_transforms(videos)
        conds = self.dataset.augment_transforms(conds)
        videos = self.dataset.img_normalize(videos)
        conds = self.dataset.img_normalize(conds)

        return videos, conds
