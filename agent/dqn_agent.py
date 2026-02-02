"""
Deep Q-Network agent with GPU optimizations.

Supports mixed precision training (AMP), torch.compile, pinned memory,
and async CPU→GPU transfers for maximum GPU utilization.
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

    GPU optimizations:
    - Mixed precision training (AMP) for ~2x speedup on tensor cores
    - torch.compile for fused kernels
    - Pinned memory replay buffer for async CPU→GPU transfers
    - Direct tensor creation on device where possible
    - set_to_none=True for zero_grad to avoid memset
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
        device: str = "auto",
        use_amp: bool = True,
        use_compile: bool = False,
        pin_memory: bool = True,
        train_steps_per_update: int = 1,
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
            use_amp: whether to use automatic mixed precision (CUDA only)
            use_compile: whether to use torch.compile (PyTorch 2.0+)
            pin_memory: whether to use pinned memory for replay buffer
            train_steps_per_update: number of gradient steps per train_step call
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
        self.train_steps_per_update = train_steps_per_update

        # Epsilon scheduling
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # AMP setup (only for CUDA)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

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

        # Compile networks for fused kernels (PyTorch 2.0+)
        if use_compile and hasattr(torch, "compile"):
            try:
                self.q_network = torch.compile(self.q_network)
                self.target_network = torch.compile(self.target_network)
            except Exception:
                pass  # Fall back to eager mode if compile fails

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay buffer with pinned memory for fast GPU transfers
        use_pin = pin_memory and self.device.type == "cuda"
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_size, pin_memory=use_pin
            )
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=buffer_size, pin_memory=use_pin
            )

        self.use_prioritized_replay = use_prioritized_replay

        # Counters
        self.training_steps = 0
        self.updates = 0

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Selects action using epsilon-greedy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        self.q_network.eval()
        with torch.no_grad():
            state = torch.as_tensor(
                observation, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.q_network(state)
        self.q_network.train()
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
        Performs one or more training steps (controlled by train_steps_per_update).

        Returns:
            Dict with metrics or None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        total_loss = 0.0
        total_q = 0.0

        for _ in range(self.train_steps_per_update):
            metrics = self._single_train_step()
            if metrics:
                total_loss += metrics["loss"]
                total_q += metrics["mean_q"]

        return {
            "loss": total_loss / self.train_steps_per_update,
            "mean_q": total_q / self.train_steps_per_update,
            "epsilon": self.epsilon,
        }

    def _single_train_step(self) -> Optional[Dict]:
        """Performs a single gradient update step."""
        # Sample batch - use direct tensor sampling for GPU path
        if self.use_prioritized_replay:
            if hasattr(self.replay_buffer, 'sample_tensors'):
                states, actions, rewards, next_states, dones, indices, weights = \
                    self.replay_buffer.sample_tensors(self.batch_size, self.device)
            else:
                states, actions, rewards, next_states, dones, indices, weights = \
                    self.replay_buffer.sample(self.batch_size)
                weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
                states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
                actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
                rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
                next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
                dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        else:
            if hasattr(self.replay_buffer, 'sample_tensors'):
                states, actions, rewards, next_states, dones = \
                    self.replay_buffer.sample_tensors(self.batch_size, self.device)
            else:
                states, actions, rewards, next_states, dones = \
                    self.replay_buffer.sample(self.batch_size)
                states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
                actions = torch.as_tensor(actions, dtype=torch.long, device=self.device)
                rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
                next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
                dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
            weights = torch.ones(self.batch_size, device=self.device)

        # Forward pass with optional AMP
        if self.use_amp:
            loss, current_q, td_errors = self._compute_loss_amp(
                states, actions, rewards, next_states, dones, weights
            )
        else:
            loss, current_q, td_errors = self._compute_loss(
                states, actions, rewards, next_states, dones, weights
            )

        # Backward pass
        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
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

    def _compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ):
        """Computes TD loss in float32."""
        # Current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        self.q_network.eval()
        self.target_network.eval()
        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q = self.target_network(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q = self.target_network(next_states).max(dim=1)[0]

            target_q = rewards + self.gamma * next_q * (1 - dones)
        self.q_network.train()
        self.target_network.train()

        td_errors = target_q - current_q
        loss = (weights * td_errors.pow(2)).mean()

        return loss, current_q, td_errors

    def _compute_loss_amp(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ):
        """Computes TD loss with automatic mixed precision."""
        with torch.amp.autocast("cuda"):
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            self.q_network.eval()
            self.target_network.eval()
            with torch.no_grad():
                if self.use_double_dqn:
                    next_actions = self.q_network(next_states).argmax(dim=1)
                    next_q = self.target_network(next_states).gather(
                        1, next_actions.unsqueeze(1)
                    ).squeeze(1)
                else:
                    next_q = self.target_network(next_states).max(dim=1)[0]

                target_q = rewards + self.gamma * next_q * (1 - dones)
            self.q_network.train()
            self.target_network.train()

            td_errors = target_q - current_q
            # Compute loss in float32 for numerical stability
            loss = (weights * td_errors.float().pow(2)).mean()

        return loss, current_q.float(), td_errors.float()

    def _update_epsilon(self):
        """Updates epsilon with linear schedule."""
        progress = min(1.0, self.training_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def save(self, path: str):
        """Saves model."""
        save_dict = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "updates": self.updates,
        }
        if self.scaler is not None:
            save_dict["scaler"] = self.scaler.state_dict()
        torch.save(save_dict, path)

    def load(self, path: str):
        """Loads model."""
        checkpoint = torch.load(path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
        self.updates = checkpoint["updates"]
        if self.scaler is not None and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
