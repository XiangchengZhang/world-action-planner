import numpy as np
from typing import List, Optional
from collections import deque
from contextlib import contextmanager
from .client import WMClient, WMPredictionOutput


class WMEnv:
    def __init__(
        self,
        env,
        empty_env,
        client: Optional[WMClient] = None,
        panel_cams: Optional[List[str]] = None,
        buffer_size: int = 250,
        warmup_frames: int = 80,
    ):
        self.env = env
        self.empty_env = empty_env
        self.client = client
        self.panel_cams = panel_cams or [
            "agentview",
            "birdview",
            "robot0_eye_in_hand",
            "sideview",
        ]
        self.buffer_size = buffer_size
        self.warmup_frames = warmup_frames
        self.history_frames = deque(maxlen=self.buffer_size)
        self.history_conds = deque(maxlen=self.buffer_size)
        self.sim_frame_buffer = None
        self.sim_cond_buffer = None

    def reset(self):
        # reset envs
        self.empty_env.reset()
        obs = self.env.reset()
        self.empty_env.copy_robot_state(self.env)

        # reset buffers
        self.history_frames = deque(maxlen=self.buffer_size)
        self.history_conds = deque(maxlen=self.buffer_size)

        for _ in range(self.warmup_frames):
            self.update_frame_buffer(obs)
            self.update_cond_buffer(copy_state=False)

        return obs

    def update_frame_buffer(self, obs):
        frames = []
        for cam in self.panel_cams:
            frame = obs[f"{cam}_image"][::-1, :, :]
            frames.append(frame)
        frame = np.vstack([np.hstack(frames[:2]), np.hstack(frames[2:])])
        self.history_frames.append(frame)

    def get_cond_frame(self):
        cond_frames = []
        cam_infos = self.empty_env.get_camera_info()
        for cam in self.panel_cams:
            cam_info = cam_infos[cam]
            height = cam_info['camera_height']
            width = cam_info['camera_width']
            cam_transform = cam_info['camera_transform']
            if "robot" in cam:
                pose_image = self.empty_env.plot_wrist_pose(cam_transform, height=height, width=width)
            else:
                pose_image = self.empty_env.plot_pose(cam_transform, height=height, width=width)
            cond_frames.append(pose_image)
        cond_frame = np.vstack([np.hstack(cond_frames[:2]), np.hstack(cond_frames[2:])])
        return cond_frame

    def update_cond_buffer(self, copy_state=True):
        if copy_state:
            self.empty_env.copy_robot_state(self.env)
        cond_frame = self.get_cond_frame()
        self.history_conds.append(cond_frame)

    def step(self, action, copy_state=True):
        obs, reward, done, info = self.env.step(action)
        self.empty_env.step(action)
        self.update_frame_buffer(obs)
        self.update_cond_buffer(copy_state=copy_state)
        return obs, reward, done, info

    @contextmanager
    def simulation(self):
        """Context manager that initializes simulation buffers on entry and cleans up on exit."""
        # Initialize simulation buffers
        self.sim_frame_buffer = list(self.history_frames)  # Start with current history frames
        self.sim_cond_buffer = list(self.history_conds)    # Start with current history conditions

        # Copy robot state on entry
        self.empty_env.copy_robot_state(self.env)

        try:
            yield self
        finally:
            # Restore robot state on exit
            self.empty_env.copy_robot_state(self.env)

            # Clean up simulation buffers
            self.sim_frame_buffer = None
            self.sim_cond_buffer = None

    def simulate(self, actions):
        if self.client is None:
            raise RuntimeError("WMEnv requires a WMClient before simulate() can be called.")
        if self.sim_frame_buffer is None or self.sim_cond_buffer is None:
            raise RuntimeError("simulate() must be called inside `with wm_env.simulation():`.")

        future_conds = []
        future_obs = []
        for i in range(len(actions)):
            obs, _, _, _ = self.empty_env.step(actions[i])
            cond_frame = self.get_cond_frame()
            future_conds.append(cond_frame)
            future_obs.append(obs)

        result = self.client.predict(
            history_frames=self.sim_frame_buffer,
            history_conds=self.sim_cond_buffer,
            future_conds=future_conds,
        )
        if not isinstance(result, WMPredictionOutput):
            message = result.get("message", "") if isinstance(result, dict) else str(result)
            exception_type = result.get("exception_type") if isinstance(result, dict) else None
            traceback = result.get("traceback") if isinstance(result, dict) else None
            details = f"{exception_type}: {message}" if exception_type else message
            if traceback:
                details = f"{details}\n{traceback}"
            raise RuntimeError(f"WM prediction failed: {details}")

        future_panels = result.pred_panels
        future_indices = np.linspace(0, len(future_conds)-1, num=len(future_panels), dtype=int)
        for i, idx in enumerate(future_indices):
            for cam_id, cam in enumerate(self.panel_cams):
                future_obs[idx][f"{cam}_image"] = future_panels[i][cam_id][::-1, :, :]

        output = {
            "WMPredictionOutput": result,
            "future_obs": [future_obs[i] for i in future_indices],
        }

        # update simulation buffer
        self.sim_cond_buffer.extend(future_conds)
        frame_indices = np.linspace(0, len(result.pred_frames)-1, num=len(future_conds), dtype=int)
        future_frames = np.array(result.pred_frames)[frame_indices]
        self.sim_frame_buffer.extend(future_frames)
        assert len(self.sim_frame_buffer) == len(self.sim_cond_buffer), (
            "Frame and condition buffers must be the same length."
        )

        return output
