import torch
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
from .wan_t2v import WanTextToVideo

import time



class WanActionTextToVideoBase(WanTextToVideo):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.max_frames = cfg.get("max_frames", self.n_frames) # total frames input, including long continuous history 
        self.hist_steps = cfg.get("hist_steps", list(range(self.cfg.hist_len))) # history steps to condition on
        self.pred_len = self.max_frames - max(self.hist_steps) - 1
        self.context_len = max(self.hist_steps) + 1  # total context length
        assert (self.hist_len - 1) % self.vae_stride[0] == 0, \
            "hist_len - 1 must be a multiple of vae_stride[0] due to temporal vae. " \
                f"Got {self.hist_len} and vae stride {self.vae_stride[0]}"
        self.hist_tokens = (self.hist_len - 1) // self.vae_stride[0] + 1
        assert len(self.hist_steps) == self.hist_len
        self.check_cfg()

    def check_cfg(self):
        assert self.diffusion_forcing.enabled, "Diffusion forcing must be enabled for conditional generation."
        assert "cond_mode" in self.diffusion_forcing, "cond_mode must be specified in diffusion_forcing."
        
    def add_training_noise(self, video_lat):
        b, _, f = video_lat.shape[:3]
        device = video_lat.device
        if self.diffusion_type == "discrete":
            video_lat = rearrange(video_lat, "b c f h w -> (b f) c h w")
            noise = torch.randn_like(video_lat)
            timesteps = self.num_train_timesteps
            if self.diffusion_forcing.enabled:
                match self.diffusion_forcing.mode:
                    case "independent":
                        t = np.random.randint(timesteps, size=(b, f))
                        if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                            t[:, 0] = timesteps - 1
                    case "rand_history":
                        # currently we aim to support two history lengths, 1 and 6
                        possible_hist_lengths = list(range(1, self.hist_tokens + 1))
                        hist_length_probs = [0.1,] + [(1 - 0.1 - 0.5) / (len(possible_hist_lengths) - 2)] * (len(possible_hist_lengths) - 2) + [0.5,]
                        t = np.zeros((b, f), dtype=np.int64)
                        for i in range(b):
                            hist_len = np.random.choice(
                                possible_hist_lengths, p=hist_length_probs
                            )
                            history_t = np.random.randint(timesteps)
                            future_t = np.random.randint(timesteps)
                            t[i, :hist_len] = history_t
                            t[i, hist_len:] = future_t
                            if (
                                np.random.rand()
                                < self.diffusion_forcing.clean_hist_prob
                            ):
                                t[i, :hist_len] = timesteps - 1
                    case "independent_history":
                        t = np.zeros((b, f), dtype=np.int64)
                        for i in range(b):
                            hist_len = self.hist_tokens
                            future_t = np.random.randint(timesteps)
                            history_t = np.random.randint(timesteps, size=(hist_len,))
                            t[i, :hist_len] = history_t
                            t[i, hist_len:] = future_t
                            if (
                                np.random.rand()
                                < self.diffusion_forcing.clean_hist_prob
                            ):
                                t[i, :hist_len] = timesteps - 1
                t = self.training_timesteps[t.flatten()].reshape(b, f)
                t_expanded = t.flatten()
            else:
                t = np.random.randint(timesteps, size=(b,))
                t_expanded = repeat(t, "b -> (b f)", f=f)
                t = self.training_timesteps[t]
                t_expanded = self.training_timesteps[t_expanded]

            noisy_lat = self.training_scheduler.add_noise(video_lat, noise, t_expanded)
            noisy_lat = rearrange(noisy_lat, "(b f) c h w -> b c f h w", b=b, f=f)
            noise = rearrange(noise, "(b f) c h w -> b c f h w", b=b, f=f)
        elif self.diffusion_type == "continuous":
            # continious time steps.
            # 1. first sample t ~ U[0, 1]
            # 2. shift t with equation: t = t * self.sample_shift / (1 + (self.sample_shift - 1) * t)
            # 3. expand t to [b, 1/f, 1, 1, 1]
            # 4. compute noisy_lat = video_lat * (1.0 - t_expanded) + noise * t_expanded
            # 5. scale t to [0, num_train_timesteps]
            # returns:
            #  t is in [0, num_train_timesteps] of shape [b, f] or [b,], of dtype torch.float32
            # video_lat is shape [b, c, f, h, w]
            # noise is shape [b, c, f, h, w]
            dist = torch.distributions.uniform.Uniform(0, 1)
            noise = torch.randn_like(video_lat)  # [b, c, f, h, w]

            if self.diffusion_forcing.enabled:
                match self.diffusion_forcing.mode:
                    case "independent":
                        t = dist.sample((b, f)).to(device)
                        if np.random.rand() < self.diffusion_forcing.clean_hist_prob:
                            t[:, 0] = 0.0
                    case "rand_history":
                        # currently we aim to support two history lengths, 1 and 6
                        possible_hist_lengths = list(range(1, self.hist_tokens + 1))
                        hist_length_probs = [0.1,] + [(1 - 0.1 - 0.5) / (len(possible_hist_lengths) - 2)] * (len(possible_hist_lengths) - 2) + [0.5,]
                        t = np.zeros((b, f), dtype=np.float32)
                        for i in range(b):
                            hist_len = np.random.choice(
                                possible_hist_lengths, p=hist_length_probs
                            )
                            history_t = np.random.uniform(0, 1)
                            future_t = np.random.uniform(0, 1)
                            t[i, :hist_len] = history_t
                            t[i, hist_len:] = future_t
                            if (
                                np.random.rand()
                                < self.diffusion_forcing.clean_hist_prob
                            ):
                                t[i, :hist_len] = 0

                        # cast dtype of t
                        t = torch.from_numpy(t).to(device)
                        t = t.float()
                    case "independent_history":
                        t = np.zeros((b, f), dtype=np.float32)
                        for i in range(b):
                            hist_len = self.hist_tokens
                            history_t = np.random.uniform(0, 1, size=(hist_len,))
                            future_t = np.random.uniform(0, 1)
                            t[i, :hist_len] = history_t
                            t[i, hist_len:] = future_t
                            if (
                                np.random.rand()
                                < self.diffusion_forcing.clean_hist_prob
                            ):
                                t[i, :hist_len] = 0
                        # cast dtype of t
                        t = torch.from_numpy(t).to(device)
                        t = t.float()
                # t is [b, f] in range [0, 1] or dtype torch.float32  0 indicates clean.
                t = t * self.sample_shift / (1 + (self.sample_shift - 1) * t)
                t_expanded = (
                    t.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # [b, f] -> [b, 1, f, 1, 1]

                # [b, c, f, h, w] * [b, 1, f, 1, 1] + [b, c, f, h, w] * [b, 1, f, 1, 1]
                noisy_lat = video_lat * (1.0 - t_expanded) + noise * t_expanded
                t = t * self.num_train_timesteps  # [b, f] -> [b, f]
                # now t is in [0, num_train_timesteps] of shape [b, f]
            else:
                t = dist.sample((b,)).to(device)
                t = t * self.sample_shift / (1 + (self.sample_shift - 1) * t)
                t_expanded = t.view(-1, 1, 1, 1, 1)

                noisy_lat = video_lat * (1.0 - t_expanded) + noise * t_expanded
                t = t * self.num_train_timesteps  # [b,]
                # now t is in [0, num_train_timesteps] of shape [b,]
        else:
            raise NotImplementedError("Unsupported time step type.")

        return noisy_lat, noise, t

    def training_step(self, batch, batch_idx=None):
        batch = self.prepare_embeds(batch)
        clip_embeds = batch["clip_embeds"]
        image_embeds = batch["image_embeds"]
        prompt_embeds = batch["prompt_embeds"]
        video_lat = batch["video_lat"]
        cond_lat = batch["cond_lat"]

        noisy_lat, noise, t = self.add_training_noise(video_lat)
        flow = noise - video_lat

        flow_pred = self.model(
            noisy_lat,
            t=t,
            context=prompt_embeds,
            clip_fea=clip_embeds,
            seq_len=self.max_tokens,
            y=image_embeds,
            cond=cond_lat,
            cond_mode=self.diffusion_forcing['cond_mode']
        )
        loss = torch.nn.functional.mse_loss(flow_pred, flow)

        if self.global_step % 100 == 0:
            print(f"[DEBUG] [Time] [{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] [Step] {self.global_step} [Cond] {self.diffusion_forcing['cond_mode']}", flush=True)
        if self.global_step % self.cfg.logging.loss_freq == 0:
            self.log("train/loss", loss, sync_dist=False, on_step=True, logger=True)

        return loss

    @torch.no_grad()
    def sample_seq(self, batch, pbar=None, **kwargs):
        """
        Main sampling loop. Only first hist_len frames are used for conditioning
        batch: dict
            batch["videos"]: [B, T, C, H, W]
            batch["prompts"]: [B]
        """
        hist_tokens = self.hist_tokens

        self.inference_scheduler, self.inference_timesteps = self.build_scheduler(False)
        lang_guidance = self.lang_guidance if self.lang_guidance else 0
        hist_guidance = self.hist_guidance if self.hist_guidance else 0

        batch = self.prepare_embeds(batch, negative_prompt=lang_guidance > 0)
        clip_embeds = batch["clip_embeds"]
        image_embeds = batch["image_embeds"]
        prompt_embeds = batch["prompt_embeds"]
        video_lat = batch["video_lat"]
        cond_lat = batch['cond_lat']
        cond_mode = self.diffusion_forcing['cond_mode']

        batch_size = video_lat.shape[0]

        video_pred_lat = torch.randn_like(video_lat)
        if self.lang_guidance:
            neg_prompt_embeds = batch["negative_prompt_embeds"]
        # if pbar is None:
        #     pbar = tqdm(range(len(self.inference_timesteps)), desc="Sampling")
        for t in self.inference_timesteps:
            if self.diffusion_forcing.enabled:
                video_pred_lat[:, :, :hist_tokens] = video_lat[:, :, :hist_tokens]
                t_expanded = torch.full((batch_size, self.lat_t), t, device=self.device)
                t_expanded[:, :hist_tokens] = self.inference_timesteps[-1]
            else:
                t_expanded = torch.full((batch_size,), t, device=self.device)

            # normal conditional sampling
            flow_pred = self.model(
                video_pred_lat,
                t=t_expanded,
                context=prompt_embeds,
                seq_len=self.max_tokens,
                clip_fea=clip_embeds,
                y=image_embeds,
                cond=cond_lat,
                cond_mode=cond_mode
            )

            # language unconditional sampling
            if lang_guidance:
                no_lang_flow_pred = self.model(
                    video_pred_lat,
                    t=t_expanded,
                    context=neg_prompt_embeds,
                    seq_len=self.max_tokens,
                    clip_fea=clip_embeds,
                    y=image_embeds,
                    cond=cond_lat,
                    cond_mode=cond_mode
                )
            else:
                no_lang_flow_pred = torch.zeros_like(flow_pred)

            # history guidance sampling:
            if hist_guidance and self.diffusion_forcing.enabled:
                print(f"[DEBUG] [hist_guidance] {hist_guidance}")
                no_hist_video_pred_lat = video_pred_lat.clone()
                no_hist_video_pred_lat[:, :, :hist_tokens] = torch.randn_like(
                    no_hist_video_pred_lat[:, :, :hist_tokens]
                )
                t_expanded[:, :hist_tokens] = self.inference_timesteps[0]
                no_hist_flow_pred = self.model(
                    no_hist_video_pred_lat,
                    t=t_expanded,
                    context=prompt_embeds,
                    seq_len=self.max_tokens,
                    clip_fea=clip_embeds,
                    y=image_embeds,
                    cond=cond_lat,
                    cond_mode=cond_mode
                )
            else:
                no_hist_flow_pred = torch.zeros_like(flow_pred)

            flow_pred = flow_pred * (1 + lang_guidance + hist_guidance)
            flow_pred = (
                flow_pred
                - lang_guidance * no_lang_flow_pred
                - hist_guidance * no_hist_flow_pred
            )

            video_pred_lat = self.remove_noise(flow_pred, t, video_pred_lat)
            # pbar.update(1)

        video_pred_lat[:, :, :hist_tokens] = video_lat[:, :, :hist_tokens]

        video_pred = rearrange(self.decode_video(video_pred_lat), "b c t h w -> b t c h w")
        video_pred = torch.concat([batch['videos'][:, :-self.pred_len], video_pred[:, -self.pred_len:]], dim=1)
        
        return video_pred
    def autoregressive_schedule(self, total_frames):
        """
        Generate an autoregressive schedule for sampling.
        Args:
            total_frames (int): Total number of frames to generate.
        Returns:
            List of tuples, each containing (start_frame, end_frame) for each step.
        """
        schedule = []
        max_frames = self.max_frames
        context_len = self.context_len
        pred_len = self.pred_len
        n_iter = np.ceil((total_frames - context_len) / pred_len)
        pred_start = context_len
        for _ in range(int(n_iter)):
            pred_end = min(pred_start + pred_len, total_frames)
            start_frame = pred_end - max_frames
            schedule.append((start_frame, pred_end))
            pred_start += pred_len
        return schedule
    
    def slice_batch(self, batch, start_frame, end_frame):
        """
        Slice the batch to only include frames from start_frame to end_frame.
        Args:
            batch (dict): Input batch containing videos and other data.
            start_frame (int): Start frame index.
            end_frame (int): End frame index.
        Returns:
            Sliced copy of the batch.
        """
        sliced_batch = {}
        tensor_keys = ['videos', 'conds']
        for key, value in batch.items():
            if key in tensor_keys:
                sliced_batch[key] = value[:, start_frame:end_frame].clone()
            else:
                sliced_batch[key] = value
        return sliced_batch
    
    def predict_seq(self, batch, **kwargs):
        """
        Autoregressive prediction of video sequences.
        Args:
            batch (dict): Input batch containing videos and other data.
        Returns:
            Generated video sequences.
        """
        total_frames = batch['videos'].shape[1]
        schedule = self.autoregressive_schedule(total_frames)
        new_batch = batch.copy()
        # only use the context frames from the input batch
        new_batch['videos'] = torch.zeros_like(batch['videos'])
        new_batch['videos'][:, :self.context_len] = batch['videos'][:, :self.context_len].clone()
        
        for step, (start_frame, end_frame) in enumerate(schedule):
            import time
            start = time.time()
            sliced_batch = self.slice_batch(new_batch, start_frame, end_frame)
            cur_video_pred = self.sample_seq(sliced_batch, **kwargs)
            # update the new_batch with the predicted frames
            new_batch['videos'][:, start_frame:end_frame] = cur_video_pred.clone()
            print(f"Sampling {start_frame}:{end_frame} took {time.time() - start:.2f} seconds.")
        return new_batch['videos']