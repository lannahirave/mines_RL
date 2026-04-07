"""
Unit tests for the Snake+ environment.
"""

from collections import deque

import pytest
import numpy as np
from env.snake_env import SnakePlusEnv
from env.snake import Direction
from env.game_objects import GameObject, ObjectType


def _neutral_reward_modifiers():
    """Disable proximity and length scaling (starvation set per test)."""
    return dict(
        proximity_good_scale=0.0,
        proximity_bad_scale=0.0,
        fruit_reward_length_coef=0.0,
        fruit_penalty_length_coef=0.0,
        fruit_penalty_min_factor=0.25,
        death_penalty_length_coef=0.0,
        death_penalty_min_scale=1.0,
    )


class TestSnakePlusEnv:
    def setup_method(self):
        self.env = SnakePlusEnv(
            grid_size=(15, 15),
            observation_type="features",
            max_steps=100,
            starvation_max_steps=-1,
            **_neutral_reward_modifiers(),
        )

    def test_reset_returns_correct_shape(self):
        obs, info = self.env.reset(seed=42)
        assert obs.shape == (24,)
        assert isinstance(info, dict)

    def test_reset_info_keys(self):
        _, info = self.env.reset(seed=42)
        assert "score" in info
        assert "length" in info
        assert "steps" in info
        assert "obstacles_count" in info
        assert "steps_since_food" in info
        assert info["steps_since_food"] == 0

    def test_step_returns_correct_tuple(self):
        self.env.reset(seed=42)
        obs, reward, terminated, truncated, info = self.env.step(0)
        assert obs.shape == (24,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_action_space(self):
        assert self.env.action_space.n == 3

    def test_observation_space_features(self):
        assert self.env.observation_space.shape == (24,)

    def test_observation_space_grid(self):
        env = SnakePlusEnv(
            grid_size=(15, 15),
            observation_type="grid",
            starvation_max_steps=-1,
            **_neutral_reward_modifiers(),
        )
        assert env.observation_space.shape == (6, 15, 15)

    def test_grid_observation(self):
        env = SnakePlusEnv(
            grid_size=(15, 15),
            observation_type="grid",
            starvation_max_steps=-1,
            **_neutral_reward_modifiers(),
        )
        obs, _ = env.reset(seed=42)
        assert obs.shape == (6, 15, 15)
        # Snake channel: head at 1.0, body segments decaying below
        assert obs[0].max() == 1.0
        assert np.count_nonzero(obs[0]) == 3  # initial snake length

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
        env = SnakePlusEnv(
            grid_size=(15, 15),
            max_steps=5,
            starvation_max_steps=-1,
            **_neutral_reward_modifiers(),
        )
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

    def test_feature_danger_order_matches_actions(self):
        """Danger [0,1,2] must align with FORWARD, LEFT, RIGHT (0,1,2)."""
        env = SnakePlusEnv(
            grid_size=(15, 15),
            observation_type="features",
            max_objects=0,
            max_steps=50,
            starvation_max_steps=-1,
            **_neutral_reward_modifiers(),
        )
        env.reset(seed=0)
        env.snake.direction = Direction.RIGHT
        env.snake.body = deque([(7, 7), (6, 7), (5, 7)])
        env.snake._body_set = None
        obs = env._get_feature_observation()
        assert obs[0] == 0.0 and obs[1] == 0.0 and obs[2] == 0.0
        env.snake.body = deque([(14, 7), (13, 7), (12, 7)])
        env.snake._body_set = None
        obs = env._get_feature_observation()
        assert obs[0] == 1.0

    def test_clear_ray_includes_body_and_obstacle(self):
        """Features 11-14 count body/obstacle along ray, not only map boundary."""
        env = SnakePlusEnv(
            grid_size=(15, 15),
            observation_type="features",
            max_objects=0,
            max_steps=50,
            starvation_max_steps=-1,
            **_neutral_reward_modifiers(),
        )
        env.reset(seed=0)
        env.snake.direction = Direction.RIGHT
        env.snake.body = deque([(5, 7), (4, 7), (3, 7)])
        env.snake._body_set = None
        env.obstacles = []
        env._obstacle_pos_set = None
        assert env._clear_steps_along_ray((1, 0)) == 9
        env.obstacles = [GameObject(8, 7, ObjectType.OBSTACLE, lifetime=-1)]
        env._invalidate_obstacle_cache()
        assert env._clear_steps_along_ray((1, 0)) == 2
        env.obstacles = []
        env._invalidate_obstacle_cache()
        env.snake.body = deque([(5, 7), (6, 7), (7, 7)])
        env.snake._body_set = None
        assert env._clear_steps_along_ray((1, 0)) == 0


class TestSnakePlusEnvGrid:
    def test_grid_channels(self):
        env = SnakePlusEnv(
            grid_size=(10, 10),
            observation_type="grid",
            starvation_max_steps=-1,
            **_neutral_reward_modifiers(),
        )
        obs, _ = env.reset(seed=42)
        # 6 channels: snake, good_prox, bad_prox, danger_prox, dir_dx, dir_dy
        assert obs.shape[0] == 6
        # All values between 0 and 1
        assert obs.min() >= 0
        assert obs.max() <= 1


class TestStarvationAndShaping:
    def test_starvation_terminates_when_no_food(self):
        env = SnakePlusEnv(
            grid_size=(15, 15),
            max_objects=0,
            max_steps=200,
            starvation_max_steps=5,
            **_neutral_reward_modifiers(),
        )
        env.reset(seed=0)
        for _ in range(4):
            _, _, term, trunc, _ = env.step(0)
            assert not term and not trunc
        _, _, term, trunc, info = env.step(0)
        assert term
        assert not trunc
        assert info["steps_since_food"] == 5

    def test_steps_since_food_increments_without_starvation_death(self):
        env = SnakePlusEnv(
            grid_size=(5, 5),
            max_objects=0,
            max_steps=200,
            starvation_max_steps=100,
            **_neutral_reward_modifiers(),
        )
        env.reset(seed=0)
        env.step(0)
        _, _, _, _, info = env.step(0)
        assert info["steps_since_food"] == 2

    def test_default_constructor_enables_starvation_and_shaping(self):
        env = SnakePlusEnv(grid_size=(5, 5), max_objects=0, max_steps=30)
        env.reset(seed=0)
        assert env.starvation_max_steps == 400
        assert env.proximity_good_scale == 0.01
        env.step(0)
        env.close()
