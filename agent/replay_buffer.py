"""
Experience Replay buffer for DQN.

Supports pinned memory for faster CPU→GPU transfers and n-step returns.
"""

import torch
import numpy as np
from collections import deque
from typing import Tuple, Optional

from .sum_tree import SumTree


class _NStepAccumulator:
    """Accumulates transitions and emits n-step returns.

    With ``n=1`` every transition is emitted immediately (no overhead).
    """

    def __init__(self, n: int, gamma: float):
        self.n = n
        self.gamma = gamma
        self._buf: deque = deque(maxlen=n)

    def push(self, state, action, reward, next_state, done):
        """Feed one transition. Returns a committed (s, a, R_n, s_n, done_n) or None."""
        self._buf.append((state, action, reward, next_state, done))

        if done:
            result = self._flush_all()
            return result

        if len(self._buf) == self.n:
            return [self._make_nstep()]

        return None

    def _make_nstep(self):
        R = 0.0
        for i in reversed(range(len(self._buf))):
            R = self._buf[i][2] + self.gamma * R * (1.0 - float(self._buf[i][4]))
        s0, a0 = self._buf[0][0], self._buf[0][1]
        last = self._buf[-1]
        return (s0, a0, R, last[3], last[4])

    def _flush_all(self):
        results = []
        while self._buf:
            results.append(self._make_nstep())
            self._buf.popleft()
        return results

    def reset(self):
        self._buf.clear()


class ReplayBuffer:
    """
    Circular buffer for storing experience.

    Uses pre-allocated numpy arrays for efficient memory access
    and supports pinned memory for faster GPU transfers.
    When ``n_step > 1``, transitions are accumulated into n-step returns
    before being committed to the buffer.
    """

    def __init__(
        self,
        capacity: int = 100000,
        pin_memory: bool = False,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        """
        Args:
            capacity: maximum buffer size
            pin_memory: if True, return pinned tensors for faster GPU transfer
            n_step: number of steps for multi-step returns (1 = standard)
            gamma: discount factor used for n-step return accumulation
        """
        self.capacity = capacity
        self.pin_memory = pin_memory and torch.cuda.is_available()
        self.position = 0
        self.size = 0
        self._initialized = False
        self._state_shape: Optional[Tuple[int, ...]] = None
        self._nstep = _NStepAccumulator(n_step, gamma) if n_step > 1 else None

    def _initialize(self, state: np.ndarray):
        """Lazily initialize storage arrays based on first observation shape."""
        self._state_shape = state.shape
        self.states = np.zeros((self.capacity, *self._state_shape), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *self._state_shape), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self._initialized = True

    def _commit(self, state, action, reward, next_state, done):
        if not self._initialized:
            self._initialize(state)
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Adds experience to buffer (accumulated into n-step returns if n>1)."""
        if self._nstep is None:
            self._commit(state, action, reward, next_state, done)
            return

        result = self._nstep.push(state, action, reward, next_state, done)
        if result is not None:
            for t in result:
                self._commit(*t)

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
            states = torch.as_tensor(self.states[indices], dtype=torch.float32).to(device)
            actions = torch.as_tensor(self.actions[indices], dtype=torch.long).to(device)
            rewards = torch.as_tensor(self.rewards[indices], dtype=torch.float32).to(device)
            next_states = torch.as_tensor(self.next_states[indices], dtype=torch.float32).to(device)
            dones = torch.as_tensor(self.dones[indices], dtype=torch.float32).to(device)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay backed by a sum tree.

    Push, sample, and priority update are all O(log n) instead of O(n).
    When ``n_step > 1``, transitions are accumulated into n-step returns.
    """

    def __init__(
        self,
        capacity: int = 100000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        pin_memory: bool = False,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        """
        Args:
            capacity: buffer size
            alpha: prioritization degree (0 = uniform, 1 = full priority)
            beta_start: initial beta value for importance sampling
            beta_frames: steps until beta = 1
            pin_memory: if True, return pinned tensors for faster GPU transfer
            n_step: number of steps for multi-step returns (1 = standard)
            gamma: discount factor used for n-step return accumulation
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.pin_memory = pin_memory and torch.cuda.is_available()

        self.tree = SumTree(capacity)
        self.position = 0
        self.size = 0
        self.frame = 0
        self._initialized = False
        self._state_shape: Optional[Tuple[int, ...]] = None
        self._nstep = _NStepAccumulator(n_step, gamma) if n_step > 1 else None

    def _initialize(self, state: np.ndarray):
        """Lazily initialize storage arrays based on first observation shape."""
        self._state_shape = state.shape
        self.states = np.zeros((self.capacity, *self._state_shape), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *self._state_shape), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self._initialized = True

    def _commit(self, state, action, reward, next_state, done):
        if not self._initialized:
            self._initialize(state)
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)
        self.tree.update(self.position, self.tree.max_priority ** self.alpha)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Adds experience with maximum priority. O(log n)."""
        if self._nstep is None:
            self._commit(state, action, reward, next_state, done)
            return

        result = self._nstep.push(state, action, reward, next_state, done)
        if result is not None:
            for t in result:
                self._commit(*t)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Samples with priority weighting. O(k log n)."""
        self.frame += 1

        beta = min(1.0, self.beta_start +
                   self.frame * (1.0 - self.beta_start) / self.beta_frames)

        indices = self.tree.sample(batch_size)

        # Importance sampling weights
        priorities = np.array([self.tree[i] for i in indices])
        min_prob = priorities.min() / self.tree.total
        weights = (priorities / self.tree.total) ** (-beta)
        max_weight = (min_prob) ** (-beta)
        weights /= max_weight

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights.astype(np.float32),
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
            t_weights = torch.from_numpy(weights).pin_memory().to(device, non_blocking=True)
        else:
            t_states = torch.as_tensor(states, dtype=torch.float32).to(device)
            t_actions = torch.as_tensor(actions, dtype=torch.long).to(device)
            t_rewards = torch.as_tensor(rewards, dtype=torch.float32).to(device)
            t_next_states = torch.as_tensor(next_states, dtype=torch.float32).to(device)
            t_dones = torch.as_tensor(dones, dtype=torch.float32).to(device)
            t_weights = torch.as_tensor(weights, dtype=torch.float32).to(device)

        return t_states, t_actions, t_rewards, t_next_states, t_dones, indices, t_weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Updates priorities. O(k log n)."""
        clipped = np.clip(np.abs(td_errors), 0, 100)
        priorities = (clipped + 1e-6) ** self.alpha
        self.tree.update_batch(indices, priorities)

    def __len__(self) -> int:
        return self.size
