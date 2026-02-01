"""
Deep Q-Network agent.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Optional

from .networks import DQN_MLP, DQN_CNN, DuelingDQN
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:
    """
    DQN agent with target network and experience replay.
    """

    def __init__(
        self,
        observation_type: str = "features",
        n_actions: int = 3,
        learning_rate: float = 1e-4,
        discount_factor: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 50000,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 1000,
        use_double_dqn: bool = True,
        use_dueling: bool = False,
        use_prioritized_replay: bool = False,
        device: str = "auto"
    ):
        """
        Args:
            observation_type: "features" or "grid"
            n_actions: number of actions
            learning_rate: learning rate
            discount_factor: gamma
            epsilon_start/end: epsilon-greedy parameters
            epsilon_decay_steps: steps for epsilon to decay to minimum
            buffer_size: replay buffer size
            batch_size: batch size
            target_update_freq: target network update frequency
            use_double_dqn: whether to use Double DQN
            use_dueling: whether to use Dueling architecture
            use_prioritized_replay: whether to use PER
            device: "cpu", "cuda", or "auto"
        """
        # Determine device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.n_actions = n_actions
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn

        # Epsilon scheduling
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # Create networks
        if observation_type == "features":
            if use_dueling:
                self.q_network = DuelingDQN(n_actions=n_actions).to(self.device)
                self.target_network = DuelingDQN(n_actions=n_actions).to(self.device)
            else:
                self.q_network = DQN_MLP(n_actions=n_actions).to(self.device)
                self.target_network = DQN_MLP(n_actions=n_actions).to(self.device)
        else:
            self.q_network = DQN_CNN(n_actions=n_actions).to(self.device)
            self.target_network = DQN_CNN(n_actions=n_actions).to(self.device)

        # Copy weights
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(capacity=buffer_size)
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_size)

        self.use_prioritized_replay = use_prioritized_replay

        # Counters
        self.training_steps = 0
        self.updates = 0

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Selects action using epsilon-greedy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_values = self.q_network(state)
            return q_values.argmax(dim=1).item()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Stores transition in buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[Dict]:
        """
        Performs one training step.

        Returns:
            Dict with metrics or None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        if self.use_prioritized_replay:
            states, actions, rewards, next_states, dones, indices, weights = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = \
                self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size).to(self.device)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: select action from q_network, evaluate with target
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q = self.target_network(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q = self.target_network(next_states).max(dim=1)[0]

            target_q = rewards + self.gamma * next_q * (1 - dones)

        # TD error
        td_errors = target_q - current_q

        # Loss (weighted for PER)
        loss = (weights * td_errors.pow(2)).mean()

        # Update network
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)

        self.optimizer.step()

        # Update priorities in PER
        if self.use_prioritized_replay:
            self.replay_buffer.update_priorities(
                indices,
                td_errors.detach().cpu().numpy()
            )

        # Update epsilon
        self._update_epsilon()

        # Update target network
        self.updates += 1
        if self.updates % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.training_steps += 1

        return {
            "loss": loss.item(),
            "mean_q": current_q.mean().item(),
            "epsilon": self.epsilon,
        }

    def _update_epsilon(self):
        """Updates epsilon with linear schedule."""
        progress = min(1.0, self.training_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def save(self, path: str):
        """Saves model."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "updates": self.updates,
        }, path)

    def load(self, path: str):
        """Loads model."""
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
        self.updates = checkpoint["updates"]
