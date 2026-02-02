"""
Unit tests for the visualization dashboard.

Note: These tests verify dashboard logic without actually rendering a window.
Pygame is initialized in headless mode for testing.
"""

import pytest
import os
import numpy as np

# Force SDL to use dummy video driver for headless testing
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from env.snake_env import SnakePlusEnv
from env.snake import Direction, Action
from agent.dqn_agent import DQNAgent
from visualization.dashboard import Dashboard


def get_test_config():
    """Returns a minimal config for testing."""
    return {
        "env": {
            "grid_size": [10, 10],
            "spawn_probs": {
                "apple": 0.50,
                "golden": 0.10,
                "poison": 0.15,
                "sour": 0.15,
                "rotten": 0.10,
            },
            "max_objects": 3,
            "obstacle_decay": 50,
            "max_steps": 50,
            "observation_type": "features",
        },
        "agent": {
            "learning_rate": 0.001,
            "discount_factor": 0.99,
            "epsilon_start": 0.0,
            "epsilon_end": 0.0,
            "epsilon_decay_steps": 1,
            "buffer_size": 1000,
            "batch_size": 32,
            "target_update_freq": 100,
            "use_double_dqn": True,
            "use_dueling": False,
            "use_prioritized_replay": False,
        },
        "training": {
            "seed": 42,
        },
    }


class TestDashboardInit:
    def test_creates_in_play_mode(self):
        config = get_test_config()
        dashboard = Dashboard(config, mode="play")
        assert dashboard.mode == "play"
        assert dashboard.agent is None
        assert dashboard.grid_size == (10, 10)

    def test_initial_stats(self):
        config = get_test_config()
        dashboard = Dashboard(config, mode="play")
        assert dashboard.episode_count == 0
        assert dashboard.best_score == 0
        assert dashboard.total_score == 0
        assert dashboard.paused is False

    def test_fps_options(self):
        config = get_test_config()
        dashboard = Dashboard(config, mode="play")
        assert len(dashboard.FPS_OPTIONS) > 0
        assert dashboard.fps_index >= 0


class TestDashboardLogic:
    def setup_method(self):
        self.config = get_test_config()
        self.dashboard = Dashboard(self.config, mode="play")

    def test_end_episode_updates_stats(self):
        self.dashboard._end_episode(10)
        assert self.dashboard.episode_count == 1
        assert self.dashboard.best_score == 10
        assert self.dashboard.total_score == 10

        self.dashboard._end_episode(20)
        assert self.dashboard.episode_count == 2
        assert self.dashboard.best_score == 20
        assert self.dashboard.total_score == 30

    def test_end_episode_tracks_history(self):
        self.dashboard._end_episode(5)
        self.dashboard._end_episode(15)
        assert self.dashboard.scores_history == [5, 15]

    def test_get_action_play_mode_default(self):
        state, _ = self.dashboard.env.reset(seed=42)
        action = self.dashboard._get_action(state)
        assert action == Action.FORWARD.value

    def test_get_action_play_mode_pending(self):
        state, _ = self.dashboard.env.reset(seed=42)
        self.dashboard._pending_action = Action.TURN_LEFT.value
        action = self.dashboard._get_action(state)
        assert action == Action.TURN_LEFT.value
        # Pending action should be consumed
        assert self.dashboard._pending_action is None

    def test_direction_to_action_forward(self):
        self.dashboard.env.reset(seed=42)
        # Snake starts facing RIGHT
        action = self.dashboard._direction_to_action(Direction.RIGHT)
        assert action == Action.FORWARD.value

    def test_direction_to_action_left_turn(self):
        self.dashboard.env.reset(seed=42)
        # Snake starts facing RIGHT, UP is a left turn
        action = self.dashboard._direction_to_action(Direction.UP)
        assert action == Action.TURN_LEFT.value

    def test_direction_to_action_right_turn(self):
        self.dashboard.env.reset(seed=42)
        # Snake starts facing RIGHT, DOWN is a right turn
        action = self.dashboard._direction_to_action(Direction.DOWN)
        assert action == Action.TURN_RIGHT.value

    def test_direction_to_action_reverse(self):
        self.dashboard.env.reset(seed=42)
        # Snake starts facing RIGHT, LEFT is reverse (180) -> forward
        action = self.dashboard._direction_to_action(Direction.LEFT)
        assert action == Action.FORWARD.value

    def test_lerp_color(self):
        c = Dashboard._lerp_color((0, 0, 0), (100, 200, 50), 0.5)
        assert c == (50, 100, 25)

    def test_lerp_color_extremes(self):
        assert Dashboard._lerp_color((10, 20, 30), (10, 20, 30), 0.0) == (10, 20, 30)
        assert Dashboard._lerp_color((0, 0, 0), (100, 100, 100), 1.0) == (100, 100, 100)


class TestDashboardWithAgent:
    def test_observe_mode_with_model(self, tmp_path):
        config = get_test_config()
        # Create and save a dummy agent
        agent = DQNAgent(
            observation_type="features",
            n_actions=3,
            epsilon_start=0.0,
            epsilon_end=0.0,
        )
        model_path = str(tmp_path / "test_model.pt")
        agent.save(model_path)

        dashboard = Dashboard(config, model_path=model_path, mode="observe")
        assert dashboard.mode == "observe"
        assert dashboard.agent is not None

    def test_observe_mode_action_selection(self, tmp_path):
        config = get_test_config()
        agent = DQNAgent(
            observation_type="features",
            n_actions=3,
            epsilon_start=0.0,
            epsilon_end=0.0,
        )
        model_path = str(tmp_path / "test_model.pt")
        agent.save(model_path)

        dashboard = Dashboard(config, model_path=model_path, mode="observe")
        state, _ = dashboard.env.reset(seed=42)
        action = dashboard._get_action(state)
        assert action in [0, 1, 2]
