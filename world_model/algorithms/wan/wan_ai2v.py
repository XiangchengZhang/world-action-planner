import torch
import torch.nn as nn
from tqdm import tqdm
from .wan_i2v import WanImageToVideo
from .wan_at2v_base import WanActionTextToVideoBase



class WanActionImageToVideo(WanActionTextToVideoBase, WanImageToVideo):
    """
    Main class for WanActionToVideo, inheriting from WanImageToVideo
    """

    def __init__(self, cfg):
        super(WanActionTextToVideo, self).__init__(cfg)
        self.max_frames = cfg.get("max_frames", self.n_frames) # total frames input, including long continuous history 
        self.hist_steps = cfg.get("hist_steps", list(range(self.cfg.hist_len))) # history steps to condition on
        self.pred_len = self.max_frames - max(self.hist_steps) - 1
        assert (self.hist_len - 1) % self.vae_stride[0] == 0, \
            "hist_len - 1 must be a multiple of vae_stride[0] due to temporal vae. " \
                f"Got {self.hist_len} and vae stride {self.vae_stride[0]}"
        self.hist_tokens = (self.hist_len - 1) // self.vae_stride[0] + 1
        assert len(self.hist_steps) == self.hist_len
        assert self.diffusion_forcing.cond_mode in ["seq", "channel"], \
            f"Unsupported cond_mode {self.diffusion_forcing.cond_mode} for WanAction"

    @torch.no_grad()
    def prepare_embeds(self, batch):
        batch = super().prepare_video_embeds(batch)
        batch = super().prepare_language_embeds(batch)
        batch = super().prepare_image_embeds(batch)

        assert batch['cond_lat'].shape == batch['video_lat'].shape
        assert self.diffusion_forcing.enabled
        
        if self.cfg.diffusion_forcing.cond_mode == "channel":
            batch["image_embeds"][:, 4:] = batch["cond_lat"]

        return batch

    def configure_model(self):
        return super(WanActionTextToVideo, self).configure_model()
    def configure_optimizers(self):
        return super(WanActionTextToVideo, self).configure_optimizers()

