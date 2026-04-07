"""
Gymnasium wrappers for observation pre-processing.
"""

import gymnasium as gym
import numpy as np
from collections import deque
from typing import Any, Optional, Tuple, Union


class FrameStack(gym.Wrapper):
    """Stacks the last ``n_frames`` observations along the channel axis.

    For grid observations the channel dimension is ``axis=0`` (C, H, W).
    For feature observations the frames are simply concatenated.

    When ``n_frames=1`` the wrapper is a transparent pass-through (no copy overhead).
    """

    def __init__(self, env: gym.Env, n_frames: int = 1):
        super().__init__(env)
        self.n_frames = n_frames

        if n_frames <= 1:
            return

        self._frames: deque = deque(maxlen=n_frames)
        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Box)
        low = obs_space.low
        high = obs_space.high

        if low.ndim == 3:
            stacked_low = np.repeat(low, n_frames, axis=0)
            stacked_high = np.repeat(high, n_frames, axis=0)
        else:
            stacked_low = np.tile(low, n_frames)
            stacked_high = np.tile(high, n_frames)

        self.observation_space = gym.spaces.Box(
            low=stacked_low, high=stacked_high, dtype=np.float32,
        )

    def _get_obs(self) -> np.ndarray:
        if self.n_frames <= 1:
            raise RuntimeError("should not be called with n_frames=1")
        frames = list(self._frames)
        if frames[0].ndim == 3:
            return np.concatenate(frames, axis=0)
        return np.concatenate(frames, axis=0)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        obs, info = self.env.reset(seed=seed, options=options)
        if self.n_frames <= 1:
            return obs, info
        for _ in range(self.n_frames):
            self._frames.append(obs)
        return self._get_obs(), info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.n_frames <= 1:
            return obs, reward, terminated, truncated, info
        self._frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info
