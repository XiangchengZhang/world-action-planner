import torch
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

# from algorithms.cogvideo import CogVideoXImageToVideo, CogVideoXVAE
from algorithms.wan import WanImageToVideo, WanTextToVideo, WanActionImageToVideo, WanActionTextToVideo
from datasets.robosuite import RobosuiteDataset
from .exp_base import BaseLightningExperiment

compatible_algorithms = dict(
    # cogvideox_i2v=CogVideoXImageToVideo,
    # cogvideox_vae=CogVideoXVAE,
    wan_i2v=WanImageToVideo,
    wan_t2v=WanTextToVideo,
    wan_toy=WanImageToVideo,
    wan_ai2v=WanActionImageToVideo,
    wan_at2v=WanActionTextToVideo,
)
compatible_datasets = dict(
        robosuite=RobosuiteDataset,
        robocasa=RobosuiteDataset,
        mimicgen=RobosuiteDataset,
        dexmimicgen=RobosuiteDataset,
        libero=RobosuiteDataset,
        libero_long=RobosuiteDataset,
    )

class VideoPredictionExperiment(BaseLightningExperiment):
    """
    A video prediction experiment
    """

    compatible_algorithms = compatible_algorithms

    compatible_datasets = compatible_datasets

    def _build_strategy(self):
        from lightning.pytorch.strategies.fsdp import FSDPStrategy

        if self.cfg.strategy == "ddp":
            return super()._build_strategy()
        elif self.cfg.strategy == "fsdp":
            if self.cfg.num_nodes >= 8:
                device_mesh = (self.cfg.num_nodes // 8, 32)
            else:
                device_mesh = (1, self.cfg.num_nodes * 4)
            return FSDPStrategy(
                mixed_precision=MixedPrecision(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.bfloat16,
                    buffer_dtype=torch.bfloat16,
                ),
                auto_wrap_policy=ModuleWrapPolicy(self.algo.classes_to_shard()),
                # sharding_strategy="FULL_SHARD",
                sharding_strategy="HYBRID_SHARD",
                device_mesh=device_mesh,
            )

        else:
            return self.cfg.strategy

    def download_dataset(self):
        dataset = self._build_dataset("training")
