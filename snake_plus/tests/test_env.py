"""
Unit tests for the Snake+ environment.
"""

import pytest
import numpy as np
from env.snake_env import SnakePlusEnv


class TestSnakePlusEnv:
    def setup_method(self):
        self.env = SnakePlusEnv(
            grid_size=(15, 15),
            observation_type="features",
            max_steps=100,
        )

    def test_reset_returns_correct_shape(self):
        obs, info = self.env.reset(seed=42)
        assert obs.shape == (18,)
        assert isinstance(info, dict)

    def test_reset_info_keys(self):
        _, info = self.env.reset(seed=42)
        assert "score" in info
        assert "length" in info
        assert "steps" in info
        assert "obstacles_count" in info

    def test_step_returns_correct_tuple(self):
        self.env.reset(seed=42)
        obs, reward, terminated, truncated, info = self.env.step(0)
        assert obs.shape == (18,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_space(self):
        assert self.env.action_space.n == 3

    def test_observation_space_features(self):
        assert self.env.observation_space.shape == (18,)

    def test_observation_space_grid(self):
        env = SnakePlusEnv(grid_size=(15, 15), observation_type="grid")
        assert env.observation_space.shape == (8, 15, 15)

    def test_grid_observation(self):
        env = SnakePlusEnv(grid_size=(15, 15), observation_type="grid")
        obs, _ = env.reset(seed=42)
        assert obs.shape == (8, 15, 15)
        # Head channel should have exactly one 1
        assert obs[0].sum() == 1.0

    def test_initial_state(self):
        _, info = self.env.reset(seed=42)
        assert info["score"] == 0
        assert info["length"] == 3
        assert info["steps"] == 0

    def test_objects_spawned(self):
        self.env.reset(seed=42)
        assert len(self.env.objects) > 0

    def test_step_increments(self):
        self.env.reset(seed=42)
        self.env.step(0)
        assert self.env.steps == 1

    def test_max_steps_truncation(self):
        env = SnakePlusEnv(grid_size=(15, 15), max_steps=5)
        env.reset(seed=42)
        for _ in range(10):
            _, _, terminated, truncated, _ = env.step(0)
            if terminated or truncated:
                break
        # Should either terminate (death) or truncate (max steps)
        assert terminated or truncated

    def test_random_agent_runs(self):
        """Test that a random agent can run through episodes."""
        obs, _ = self.env.reset(seed=42)
        total_steps = 0
        for _ in range(100):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_steps += 1
            if terminated or truncated:
                break
        assert total_steps > 0


class TestSnakePlusEnvGrid:
    def test_grid_channels(self):
        env = SnakePlusEnv(grid_size=(10, 10), observation_type="grid")
        obs, _ = env.reset(seed=42)
        # 8 channels
        assert obs.shape[0] == 8
        # All values between 0 and 1
        assert obs.min() >= 0
        assert obs.max() <= 1
