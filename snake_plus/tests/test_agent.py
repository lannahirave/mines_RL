"""
Unit tests for agents.
"""

import pytest
import numpy as np
from agent.q_table_agent import QTableAgent
from agent.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class TestQTableAgent:
    def setup_method(self):
        self.agent = QTableAgent(n_actions=3)

    def test_select_action_returns_valid(self):
        obs = np.random.rand(18).astype(np.float32)
        action = self.agent.select_action(obs)
        assert 0 <= action < 3

    def test_discretize_state(self):
        obs = np.array([0.0, 0.6, 0.3, 1.0] + [0.0] * 14, dtype=np.float32)
        state = self.agent.discretize_state(obs)
        assert state[0] == 0
        assert state[1] == 1
        assert state[2] == 0
        assert state[3] == 1

    def test_update_returns_td_error(self):
        obs = np.random.rand(18).astype(np.float32)
        next_obs = np.random.rand(18).astype(np.float32)
        td_error = self.agent.update(obs, 0, 1.0, next_obs, False)
        assert isinstance(td_error, float)

    def test_epsilon_decay(self):
        obs = np.random.rand(18).astype(np.float32)
        initial_eps = self.agent.epsilon
        self.agent.update(obs, 0, 1.0, obs, False)
        assert self.agent.epsilon < initial_eps

    def test_q_table_grows(self):
        assert len(self.agent.q_table) == 0
        obs = np.random.rand(18).astype(np.float32)
        # Use training=False to force greedy (accesses q_table)
        self.agent.select_action(obs, training=False)
        assert len(self.agent.q_table) >= 1


class TestReplayBuffer:
    def setup_method(self):
        self.buffer = ReplayBuffer(capacity=100)

    def test_push_and_len(self):
        state = np.zeros(18)
        self.buffer.push(state, 0, 1.0, state, False)
        assert len(self.buffer) == 1

    def test_sample(self):
        state = np.zeros(18)
        for i in range(10):
            self.buffer.push(state, i % 3, float(i), state, False)

        states, actions, rewards, next_states, dones = self.buffer.sample(5)
        assert states.shape == (5, 18)
        assert actions.shape == (5,)
        assert rewards.shape == (5,)

    def test_capacity(self):
        state = np.zeros(18)
        for i in range(200):
            self.buffer.push(state, 0, 0.0, state, False)
        assert len(self.buffer) == 100


class TestPrioritizedReplayBuffer:
    def setup_method(self):
        self.buffer = PrioritizedReplayBuffer(capacity=100)

    def test_push_and_len(self):
        state = np.zeros(18)
        self.buffer.push(state, 0, 1.0, state, False)
        assert len(self.buffer) == 1

    def test_sample_returns_weights(self):
        state = np.zeros(18)
        for i in range(10):
            self.buffer.push(state, 0, float(i), state, False)

        result = self.buffer.sample(5)
        assert len(result) == 7  # states, actions, rewards, next_states, dones, indices, weights

    def test_update_priorities(self):
        state = np.zeros(18)
        for i in range(10):
            self.buffer.push(state, 0, float(i), state, False)

        _, _, _, _, _, indices, _ = self.buffer.sample(5)
        td_errors = np.ones(5)
        self.buffer.update_priorities(indices, td_errors)
        # Should not raise
