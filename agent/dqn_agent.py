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
import torchinfo

from .networks import DQN_MLP, DQN_CNN, DQN_CNN_Shallow
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


def _unwrap_compiled_network(module: nn.Module) -> nn.Module:
    """Return the inner module when ``module`` is a torch.compile() wrapper."""
    return getattr(module, "_orig_mod", module)


def _strip_compile_state_dict_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    torch.compile checkpoints often prefix parameter names with ``_orig_mod.``.
    Normalize so the same file loads into eager or compiled networks.
    """
    prefix = "_orig_mod."
    if not state_dict or not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}


def _load_network_weights(net: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    state_dict = _strip_compile_state_dict_prefix(state_dict)
    _unwrap_compiled_network(net).load_state_dict(state_dict)


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
        lr_cosine_steps: int = 500000,
        lr_min: float = 1e-6,
        warmup_steps: int = 0,
        tau: float = 1.0,
        sign_log_reward: bool = False,
        n_step: int = 1,
        n_frames: int = 1,
        grid_size: tuple = (15, 15),
        feature_size: int = 29,
        network_type: str = "grid",
        per_beta_frames: int = 100000,
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
            target_update_freq: target network update frequency (used only when tau=1.0)
            use_double_dqn: whether to use Double DQN
            use_dueling: whether to use Dueling architecture
            use_prioritized_replay: whether to use PER
            device: "cpu", "cuda", "mps", or "auto"
            use_amp: whether to use automatic mixed precision (CUDA only)
            use_compile: whether to use torch.compile (PyTorch 2.0+)
            pin_memory: whether to use pinned memory for replay buffer
            train_steps_per_update: number of gradient steps per train_step call
            lr_cosine_steps: T_max for cosine annealing (total training steps)
            lr_min: minimum learning rate at the end of cosine schedule
            warmup_steps: minimum buffer size before training begins
            tau: target network interpolation (1.0 = hard copy, <1 = Polyak averaging)
            sign_log_reward: if True, use sign(r)*log(1+|r|) reward scaling
            n_step: number of steps for multi-step returns (1 = standard)
            n_frames: number of stacked frames (multiplies input channels/features)
            grid_size: (H, W) of the grid observation
            feature_size: base feature vector length (before frame stacking)
            network_type: CNN architecture ("grid" for deep, "grid_shallow" for shallow)
            per_beta_frames: steps over which PER beta anneals from beta_start to 1.0
        """
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.n_actions = n_actions
        self.gamma = discount_factor
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.train_steps_per_update = train_steps_per_update
        self.warmup_steps = max(warmup_steps, batch_size)
        self.tau = tau
        self.sign_log_reward = sign_log_reward
        self.n_step = n_step
        self.n_step_gamma = discount_factor ** n_step

        # Epsilon scheduling
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps

        # AMP setup (only for CUDA)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # Create networks (input dimensions account for frame stacking)
        self.observation_type = observation_type
        _CNN_CLASSES = {
            "grid": DQN_CNN,
            "grid_shallow": DQN_CNN_Shallow,
        }
        if observation_type == "features":
            mlp_input = feature_size * n_frames
            input_size = (self.batch_size, mlp_input)
            self.q_network = DQN_MLP(input_size=mlp_input, n_actions=n_actions, use_dueling=use_dueling).to(self.device)
            self.target_network = DQN_MLP(input_size=mlp_input, n_actions=n_actions, use_dueling=use_dueling).to(self.device)
        else:
            cnn_channels = 6 * n_frames
            input_size = (self.batch_size, cnn_channels, *grid_size)
            cnn_cls = _CNN_CLASSES.get(network_type, DQN_CNN)
            self.q_network = cnn_cls(input_channels=cnn_channels, grid_size=grid_size, n_actions=n_actions, use_dueling=use_dueling).to(self.device)
            self.target_network = cnn_cls(input_channels=cnn_channels, grid_size=grid_size, n_actions=n_actions, use_dueling=use_dueling).to(self.device)

        print("Q-network:")
        print(torchinfo.summary(self.q_network, input_size=input_size, device=self.device))
        print("*" * 100)
        print("Target network:")
        print(torchinfo.summary(self.target_network, input_size=input_size, device=self.device))
        print("*" * 100)

        # Copy weights
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Compile networks for fused kernels (PyTorch 2.0+)
        if use_compile and hasattr(torch, "compile"):
            try:
                self.q_network = torch.compile(self.q_network)
                self.target_network = torch.compile(self.target_network)
            except Exception:
                pass  # Fall back to eager mode if compile fails

        # Optimizer + cosine LR scheduler
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=lr_cosine_steps,
            eta_min=lr_min,
        )

        # Replay buffer with pinned memory for fast GPU transfers
        use_pin = pin_memory and self.device.type == "cuda"
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                capacity=buffer_size, pin_memory=use_pin,
                n_step=n_step, gamma=discount_factor,
                beta_frames=per_beta_frames,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                capacity=buffer_size, pin_memory=use_pin,
                n_step=n_step, gamma=discount_factor,
            )

        self.use_prioritized_replay = use_prioritized_replay

        # Counters
        self.training_steps = 0
        self.updates = 0

    def select_action(self, observation: np.ndarray, training: bool = True) -> int:
        """Selects action using epsilon-greedy."""
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        with torch.no_grad():
            # NumPy arrays + as_tensor(..., device=cuda) can stay on CPU; .to() is reliable.
            state = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
            state = state.to(self.device, non_blocking=self.device.type == "cuda")
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
        Performs one or more training steps (controlled by train_steps_per_update).

        Returns:
            Dict with metrics or None if buffer too small or still warming up
        """
        if len(self.replay_buffer) < self.warmup_steps:
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
                weights = torch.as_tensor(weights, dtype=torch.float32).to(self.device)
                states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
                actions = torch.as_tensor(actions, dtype=torch.long).to(self.device)
                rewards = torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
                next_states = torch.as_tensor(next_states, dtype=torch.float32).to(self.device)
                dones = torch.as_tensor(dones, dtype=torch.float32).to(self.device)
        else:
            if hasattr(self.replay_buffer, 'sample_tensors'):
                states, actions, rewards, next_states, dones = \
                    self.replay_buffer.sample_tensors(self.batch_size, self.device)
            else:
                states, actions, rewards, next_states, dones = \
                    self.replay_buffer.sample(self.batch_size)
                states = torch.as_tensor(states, dtype=torch.float32).to(self.device)
                actions = torch.as_tensor(actions, dtype=torch.long).to(self.device)
                rewards = torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
                next_states = torch.as_tensor(next_states, dtype=torch.float32).to(self.device)
                dones = torch.as_tensor(dones, dtype=torch.float32).to(self.device)
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
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
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
        if self.tau < 1.0:
            q_params = self.q_network.state_dict()
            t_params = self.target_network.state_dict()
            for key in t_params:
                t_params[key] = self.tau * q_params[key] + (1.0 - self.tau) * t_params[key]
            self.target_network.load_state_dict(t_params)
        elif self.updates % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.training_steps += 1
        self.lr_scheduler.step()

        return {
            "loss": loss.item(),
            "mean_q": current_q.mean().item(),
            "epsilon": self.epsilon,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

    def _scale_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.sign_log_reward:
            return torch.sign(rewards) * torch.log1p(torch.abs(rewards))
        return rewards

    def _compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ):
        """Computes TD loss in float32 using Huber loss."""
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_double_dqn:
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q = self.target_network(next_states).gather(
                    1, next_actions.unsqueeze(1)
                ).squeeze(1)
            else:
                next_q = self.target_network(next_states).max(dim=1)[0]

            scaled_rewards = self._scale_rewards(rewards)
            target_q = scaled_rewards + self.n_step_gamma * next_q * (1 - dones)

        td_errors = target_q - current_q
        loss = (weights * nn.functional.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

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
        """Computes TD loss with automatic mixed precision using Huber loss."""
        with torch.amp.autocast("cuda"):
            current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                if self.use_double_dqn:
                    next_actions = self.q_network(next_states).argmax(dim=1)
                    next_q = self.target_network(next_states).gather(
                        1, next_actions.unsqueeze(1)
                    ).squeeze(1)
                else:
                    next_q = self.target_network(next_states).max(dim=1)[0]

                scaled_rewards = self._scale_rewards(rewards)
                target_q = scaled_rewards + self.n_step_gamma * next_q * (1 - dones)

            td_errors = (target_q - current_q).float()
            loss = (weights * nn.functional.smooth_l1_loss(
                current_q.float(), target_q.float(), reduction='none'
            )).mean()

        return loss, current_q.float(), td_errors

    def _update_epsilon(self):
        """Updates epsilon with linear schedule."""
        progress = min(1.0, self.training_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def save(self, path: str):
        """Saves model."""
        save_dict = {
            "q_network": _unwrap_compiled_network(self.q_network).state_dict(),
            "target_network": _unwrap_compiled_network(self.target_network).state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps,
            "updates": self.updates,
            "n_step": self.n_step,
            "tau": self.tau,
            "sign_log_reward": self.sign_log_reward,
        }
        if self.scaler is not None:
            save_dict["scaler"] = self.scaler.state_dict()
        torch.save(save_dict, path)

    def load(self, path: str):
        """Loads model."""
        checkpoint = torch.load(path, map_location=self.device)

        _load_network_weights(self.q_network, checkpoint["q_network"])
        _load_network_weights(self.target_network, checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.epsilon = checkpoint["epsilon"]
        self.training_steps = checkpoint["training_steps"]
        self.updates = checkpoint["updates"]
        if self.scaler is not None and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
