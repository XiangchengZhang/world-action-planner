from typing import List
from torch.utils.data import IterableDataset, Dataset
from omegaconf import DictConfig
import torch
import numpy as np
from datasets.dummy import DummyVideoDataset
from datasets.openx_base import OpenXVideoDataset
from datasets.droid import DroidVideoDataset
from datasets.something_something import SomethingSomethingDataset
from datasets.epic_kitchen import EpicKitchenDataset
from datasets.pandas import PandasVideoDataset
from datasets.deprecated.video_1x_wm import WorldModel1XDataset
from datasets.agibot_world import AgibotWorldDataset
from datasets.ego4d import Ego4DVideoDataset
from datasets.robosuite import RobosuiteDataset

subset_classes = dict(
    dummy=DummyVideoDataset,
    something_something=SomethingSomethingDataset,
    epic_kitchen=EpicKitchenDataset,
    pandas=PandasVideoDataset,
    agibot_world=AgibotWorldDataset,
    video_1x_wm=WorldModel1XDataset,
    ego4d=Ego4DVideoDataset,
    droid=DroidVideoDataset,
    austin_buds=OpenXVideoDataset,
    austin_sailor=OpenXVideoDataset,
    austin_sirius=OpenXVideoDataset,
    bc_z=OpenXVideoDataset,
    berkeley_autolab=OpenXVideoDataset,
    berkeley_cable=OpenXVideoDataset,
    berkeley_fanuc=OpenXVideoDataset,
    bridge=OpenXVideoDataset,
    cmu_stretch=OpenXVideoDataset,
    dlr_edan=OpenXVideoDataset,
    dobbe=OpenXVideoDataset,
    fmb=OpenXVideoDataset,
    fractal=OpenXVideoDataset,
    iamlab_cmu=OpenXVideoDataset,
    jaco_play=OpenXVideoDataset,
    language_table=OpenXVideoDataset,
    nyu_franka=OpenXVideoDataset,
    roboturk=OpenXVideoDataset,
    stanford_hydra=OpenXVideoDataset,
    taco_play=OpenXVideoDataset,
    toto=OpenXVideoDataset,
    ucsd_kitchen=OpenXVideoDataset,
    utaustin_mutex=OpenXVideoDataset,
    viola=OpenXVideoDataset,
    robosuite=RobosuiteDataset,
    robocasa=RobosuiteDataset,
    mimicgen=RobosuiteDataset,
    dexmimicgen=RobosuiteDataset,
)


class MixtureDataset(IterableDataset):
    """
    A fault tolerant mixture of video datasets
    """

    def __init__(self, cfg: DictConfig, split: str = "training"):
        super().__init__()
        self.cfg = cfg
        self.debug = cfg.debug
        self.split = split
        self.random_seed = np.random.get_state()[1][0]  # Get current numpy random seed
        self.subset_cfg = {
            k.split("/")[1]: v for k, v in self.cfg.items() if k.startswith("subset/")
        }
        if split == "all":
            raise ValueError("split cannot be `all` for MixtureDataset`")
        weight = dict(self.cfg[split].weight)
        # Check if all keys in weight exist in subset_cfg
        for key in weight:
            if key not in self.subset_cfg:
                raise ValueError(
                    f"Dataset '{key}' specified in weights but not found in configuration"
                )
        self.subset_cfg = {k: v for k, v in self.subset_cfg.items() if k in weight}
        weight_type = self.cfg[split].weight_type  # one of relative or absolute
        self.subsets: List[Dataset] = []
        for subset_name, subset_cfg in self.subset_cfg.items():
            subset_cfg["height"] = self.cfg.height
            subset_cfg["width"] = self.cfg.width
            subset_cfg["n_frames"] = self.cfg.n_frames
            subset_cfg["fps"] = self.cfg.fps
            subset_cfg["load_video_latent"] = self.cfg.load_video_latent
            subset_cfg["load_prompt_embed"] = self.cfg.load_prompt_embed
            subset_cfg["max_text_tokens"] = self.cfg.max_text_tokens
            subset_cfg["image_to_video"] = self.cfg.image_to_video
            self.subsets.append(subset_classes[subset_name](subset_cfg, split))
            if weight_type == "relative":
                weight[subset_name] = weight[subset_name] * len(self.subsets[-1])

        # Normalize weights to sum to 1
        total_weight = sum(weight.values())
        self.normalized_weights = {k: v / total_weight for k, v in weight.items()}

        # Store dataset sizes for printing
        dataset_sizes = {
            subset_name: len(subset)
            for subset_name, subset in zip(self.subset_cfg.keys(), self.subsets)
        }

        # Print normalized weights and dataset sizes in a nice format
        print("\nDataset information for split '{}':".format(self.split))
        print("-" * 60)
        print(f"{'Dataset':<25} {'Size':<10} {'Weight':<10} {'Normalized':<10}")
        print("-" * 60)
        for subset_name, norm_weight in sorted(
            self.normalized_weights.items(), key=lambda x: -x[1]
        ):
            size = dataset_sizes[subset_name]
            orig_weight = self.cfg[split].weight[subset_name]
            print(
                f"{subset_name:<25} {size:<10,d} {orig_weight:<10.4f} {norm_weight:<10.4f}"
            )
        print("-" * 60)

        # Calculate cumulative probabilities for sampling
        self.cumsum_weights = {}
        cumsum = 0
        for k, v in self.normalized_weights.items():
            cumsum += v
            self.cumsum_weights[k] = cumsum

        # some scripts want to access the records
        self.records = []
        for subset in self.subsets:
            self.records.extend(subset.records)

    def __iter__(self):
        while True:
            # Sample a random subset based on weights using numpy random
            rand = np.random.random()
            for subset_name, cumsum in self.cumsum_weights.items():
                if rand <= cumsum:
                    selected_subset = subset_name
                    break

            # Get the corresponding dataset index
            subset_idx = list(self.subset_cfg.keys()).index(selected_subset)

            try:
                # Sample randomly from the selected dataset using numpy random
                dataset = self.subsets[subset_idx]
                idx = np.random.randint(len(dataset))
                sample = dataset[idx]
                yield sample
            except Exception as e:
                if self.debug:
                    raise e
                else:
                    print(f"Error sampling from {selected_subset}: {str(e)}")
                    continue
