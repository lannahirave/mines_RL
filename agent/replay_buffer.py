"""
Experience Replay buffer for DQN.

Supports pinned memory for faster CPU→GPU transfers.
"""

import torch
import numpy as np
from typing import Tuple, Optional
import random


class ReplayBuffer:
    """
    Circular buffer for storing experience.

    Uses pre-allocated numpy arrays for efficient memory access
    and supports pinned memory for faster GPU transfers.
    """

    def __init__(self, capacity: int = 100000, pin_memory: bool = False):
        """
        Args:
            capacity: maximum buffer size
            pin_memory: if True, return pinned tensors for faster GPU transfer
        """
        self.capacity = capacity
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.position = 0
        self.size = 0
        self._initialized = False
        self._state_shape: Optional[Tuple[int, ...]] = None

    def _initialize(self, state: np.ndarray):
        """Lazily initialize storage arrays based on first observation shape."""
        self._state_shape = state.shape
        self.states = np.zeros((self.capacity, *self._state_shape), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *self._state_shape), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self._initialized = True

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Adds experience to buffer."""
        if not self._initialized:
            self._initialize(state)

        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Randomly samples batch_size elements.

        Returns:
            (states, actions, rewards, next_states, dones)
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def sample_tensors(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """
        Samples batch and returns GPU-ready tensors directly.

        Uses pinned memory for async CPU→GPU transfers when available.

        Args:
            batch_size: number of samples
            device: target torch device

        Returns:
            (states, actions, rewards, next_states, dones) as tensors on device
        """
        indices = np.random.randint(0, self.size, size=batch_size)

        if self.pin_memory and device.type == "cuda":
            states = torch.from_numpy(self.states[indices]).pin_memory().to(device, non_blocking=True)
            actions = torch.from_numpy(self.actions[indices]).pin_memory().to(device, non_blocking=True)
            rewards = torch.from_numpy(self.rewards[indices]).pin_memory().to(device, non_blocking=True)
            next_states = torch.from_numpy(self.next_states[indices]).pin_memory().to(device, non_blocking=True)
            dones = torch.from_numpy(self.dones[indices]).pin_memory().to(device, non_blocking=True)
        else:
            states = torch.as_tensor(self.states[indices], dtype=torch.float32, device=device)
            actions = torch.as_tensor(self.actions[indices], dtype=torch.long, device=device)
            rewards = torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device)
            next_states = torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=device)
            dones = torch.as_tensor(self.dones[indices], dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay.
    Experience with higher TD error is sampled more frequently.

    Uses pre-allocated arrays and supports pinned memory for GPU transfers.
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        pin_memory: bool = False
    ):
        """
        Args:
            capacity: buffer size
            alpha: prioritization degree (0 = uniform, 1 = full priority)
            beta_start: initial beta value for importance sampling
            beta_frames: steps until beta = 1
            pin_memory: if True, return pinned tensors for faster GPU transfer
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.pin_memory = pin_memory and torch.cuda.is_available()

        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        self.frame = 0
        self._initialized = False
        self._state_shape: Optional[Tuple[int, ...]] = None

    def _initialize(self, state: np.ndarray):
        """Lazily initialize storage arrays based on first observation shape."""
        self._state_shape = state.shape
        self.states = np.zeros((self.capacity, *self._state_shape), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *self._state_shape), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self._initialized = True

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Adds experience with maximum priority."""
        if not self._initialized:
            self._initialize(state)

        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0

        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        self.priorities[self.position] = max_priority

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Samples with priority weighting."""
        self.frame += 1

        # Compute beta
        beta = min(1.0, self.beta_start +
                   self.frame * (1.0 - self.beta_start) / self.beta_frames)

        # Probabilities
        priorities = self.priorities[:self.size]
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # Sample indices
        indices = np.random.choice(self.size, batch_size, p=probs)

        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights,
        )

    def sample_tensors(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        """
        Samples batch and returns GPU-ready tensors directly.

        Args:
            batch_size: number of samples
            device: target torch device

        Returns:
            (states, actions, rewards, next_states, dones, indices, weights) as tensors
        """
        states, actions, rewards, next_states, dones, indices, weights = self.sample(batch_size)

        if self.pin_memory and device.type == "cuda":
            t_states = torch.from_numpy(states).pin_memory().to(device, non_blocking=True)
            t_actions = torch.from_numpy(actions).pin_memory().to(device, non_blocking=True)
            t_rewards = torch.from_numpy(rewards).pin_memory().to(device, non_blocking=True)
            t_next_states = torch.from_numpy(next_states).pin_memory().to(device, non_blocking=True)
            t_dones = torch.from_numpy(dones).pin_memory().to(device, non_blocking=True)
            t_weights = torch.from_numpy(weights.astype(np.float32)).pin_memory().to(device, non_blocking=True)
        else:
            t_states = torch.as_tensor(states, dtype=torch.float32, device=device)
            t_actions = torch.as_tensor(actions, dtype=torch.long, device=device)
            t_rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device)
            t_next_states = torch.as_tensor(next_states, dtype=torch.float32, device=device)
            t_dones = torch.as_tensor(dones, dtype=torch.float32, device=device)
            t_weights = torch.as_tensor(weights, dtype=torch.float32, device=device)

        return t_states, t_actions, t_rewards, t_next_states, t_dones, indices, t_weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Updates priorities."""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6

    def __len__(self) -> int:
        return self.size
