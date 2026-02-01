"""
Neural networks for DQN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class DQN_MLP(nn.Module):
    """
    Multi-layer perceptron for feature-based observations.
    """

    def __init__(
        self,
        input_size: int = 18,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        n_actions: int = 3
    ):
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, n_actions))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DQN_CNN(nn.Module):
    """
    CNN for grid-based observations.
    """

    def __init__(
        self,
        input_channels: int = 8,
        grid_size: Tuple[int, int] = (15, 15),
        n_actions: int = 3
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # Compute size after conv
        conv_out_size = self._get_conv_output_size(input_channels, grid_size)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions)
        )

    def _get_conv_output_size(self, channels: int, grid_size: Tuple[int, int]) -> int:
        dummy = torch.zeros(1, channels, grid_size[1], grid_size[0])
        out = self.conv(dummy)
        return int(np.prod(out.shape[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(x)


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture.
    Q(s, a) = V(s) + A(s, a) - mean(A(s, a'))
    """

    def __init__(
        self,
        input_size: int = 18,
        hidden_size: int = 128,
        n_actions: int = 3
    ):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )

        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)

        value = self.value(features)
        advantage = self.advantage(features)

        # Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values
