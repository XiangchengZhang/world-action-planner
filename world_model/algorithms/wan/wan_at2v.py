import torch
from einops import rearrange
from .wan_at2v_base import WanActionTextToVideoBase


class WanActionTextToVideo(WanActionTextToVideoBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.check_cfg()

    def check_cfg(self):
        super().check_cfg()
        assert self.diffusion_forcing.cond_mode == "concat", \
            "Pose conditioning must use concat cond_mode."
        
    @torch.no_grad()
    def prepare_video_embeds(self, batch, **kwargs):
        videos = batch["videos"]
        conds = batch["conds"]
        batch_size, t, _, h, w = videos.shape

        if t != self.max_frames:
            raise ValueError(f"Number of frames in videos must be {self.max_frames}")
        if h != self.height or w != self.width:
            raise ValueError(
                f"Height and width of videos must be {self.height} and {self.width}"
            )

        # get sparse history frames
        indices = list(self.hist_steps) + list(range(self.max_frames - self.pred_len, self.max_frames))
        assert len(indices) == self.n_frames, \
            f"Total selected frames {len(indices)} not equal to model n_frames {self.n_frames}"
        videos = videos[:, indices]
        conds = conds[:, indices]

        video_lat = self.encode_video(rearrange(videos, "b t c h w -> b c t h w"))
        # video_lat ~ (b, lat_c, lat_t, lat_h, lat_w)
        batch["video_lat"] = video_lat

        cond_lat = self.encode_video(rearrange(conds, "b t c h w -> b c t h w"))        
        batch["cond_lat"] = cond_lat

        return batch