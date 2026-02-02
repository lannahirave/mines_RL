"""
Unit tests for the experiments module.
"""

import pytest
import numpy as np
import yaml
from pathlib import Path
from unittest.mock import patch

from env.snake_env import SnakePlusEnv
from agent.dqn_agent import DQNAgent
from experiments.discount_analysis import (
    create_env,
    train_with_gamma,
    evaluate_agent,
    run_experiment,
    set_seed,
)


def get_test_config():
    """Returns a minimal config for fast testing."""
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
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay_steps": 500,
            "buffer_size": 1000,
            "batch_size": 32,
            "target_update_freq": 100,
            "use_double_dqn": True,
            "use_dueling": False,
            "use_prioritized_replay": False,
        },
        "training": {
            "n_episodes": 10,
            "eval_freq": 5,
            "save_freq": 5,
            "seed": 42,
        },
    }


class TestCreateEnv:
    def test_creates_valid_env(self):
        config = get_test_config()
        env = create_env(config)
        assert isinstance(env, SnakePlusEnv)
        assert env.grid_size == (10, 10)
        env.close()

    def test_env_can_reset_and_step(self):
        config = get_test_config()
        env = create_env(config)
        obs, info = env.reset()
        assert obs.shape == (18,)
        obs, reward, term, trunc, info = env.step(0)
        assert isinstance(reward, float)
        env.close()


class TestSetSeed:
    def test_reproducibility(self):
        set_seed(42)
        a = np.random.random(5)
        set_seed(42)
        b = np.random.random(5)
        np.testing.assert_array_equal(a, b)


class TestEvaluateAgent:
    def setup_method(self):
        self.config = get_test_config()
        self.agent = DQNAgent(
            observation_type="features",
            n_actions=3,
            epsilon_start=0.0,
            epsilon_end=0.0,
        )

    def test_returns_expected_keys(self):
        results = evaluate_agent(self.agent, self.config, n_episodes=3)
        assert "mean_score" in results
        assert "std_score" in results
        assert "mean_survival_steps" in results
        assert "mean_final_length" in results
        assert "survival_rate" in results
        assert "obstacle_death_rate" in results
        assert "death_causes" in results

    def test_death_causes_length(self):
        results = evaluate_agent(self.agent, self.config, n_episodes=5)
        assert len(results["death_causes"]) == 5

    def test_rates_are_valid(self):
        results = evaluate_agent(self.agent, self.config, n_episodes=5)
        assert 0 <= results["survival_rate"] <= 1
        assert 0 <= results["obstacle_death_rate"] <= 1


class TestTrainWithGamma:
    def test_returns_metrics(self, tmp_path):
        config = get_test_config()
        result = train_with_gamma(config, gamma=0.5, run_dir=tmp_path, n_episodes=5)
        assert result["gamma"] == 0.5
        assert len(result["episode_rewards"]) == 5
        assert len(result["episode_scores"]) == 5
        assert "eval" in result

    def test_saves_model(self, tmp_path):
        config = get_test_config()
        train_with_gamma(config, gamma=0.99, run_dir=tmp_path, n_episodes=3)
        assert (tmp_path / "model_gamma_0.99.pt").exists()


class TestRunExperiment:
    def test_full_experiment(self, tmp_path):
        config = get_test_config()
        results = run_experiment(
            config, gammas=[0.5, 0.99], n_episodes=3, output_dir=tmp_path
        )
        assert len(results) == 2
        assert results[0]["gamma"] == 0.5
        assert results[1]["gamma"] == 0.99

        # Check output files
        assert (tmp_path / "experiment_config.yaml").exists()
        assert (tmp_path / "summary.yaml").exists()
        assert (tmp_path / "training_curves_comparison.png").exists()
        assert (tmp_path / "evaluation_comparison.png").exists()
