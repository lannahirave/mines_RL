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

    With ``use_dueling=True`` the final hidden representation is split
    into a value stream V(s) and an advantage stream A(s, a), combined
    as Q(s, a) = V(s) + A(s, a) - mean(A).
    """

    def __init__(
        self,
        input_size: int = 29,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        n_actions: int = 3,
        use_dueling: bool = False,
    ):
        super().__init__()
        self.use_dueling = use_dueling

        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU()])
            prev_size = hidden_size
        self.backbone = nn.Sequential(*layers)

        if use_dueling:
            self.value_head = nn.Sequential(
                nn.Linear(prev_size, prev_size), nn.ReLU(), nn.Linear(prev_size, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(prev_size, prev_size), nn.ReLU(), nn.Linear(prev_size, n_actions),
            )
        else:
            self.q_head = nn.Linear(prev_size, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if self.use_dueling:
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.q_head(features)


class _ResBlock(nn.Module):
    """Residual block with GroupNorm (acts as LayerNorm for conv layers)."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.relu(out + identity)


class DQN_CNN(nn.Module):
    """
    CNN for grid-based observations on small grids (e.g. 15x15).

    Architecture:
    - CoordConv: appends normalized (x, y) coordinate channels for
      absolute position awareness.
    - BatchNorm2d after the first convolution for observation normalization.
    - Strided convolutions reduce spatial dims (15x15 -> 8x8 -> 4x4)
      instead of global pooling, preserving relative spatial relationships.
    - Residual blocks after each spatial scale for deeper feature extraction.
    - Flatten (64*4*4 = 1024) -> FC -> dueling or plain Q heads.
    """

    def __init__(
        self,
        input_channels: int = 6,
        grid_size: Tuple[int, int] = (15, 15),
        n_actions: int = 3,
        use_dueling: bool = True,
    ):
        super().__init__()
        self.use_dueling = use_dueling
        d0, d1 = grid_size

        coord0 = torch.linspace(-1, 1, d0).view(1, 1, d0, 1).expand(1, 1, d0, d1)
        coord1 = torch.linspace(-1, 1, d1).view(1, 1, 1, d1).expand(1, 1, d0, d1)
        self.register_buffer("_coords", torch.cat([coord0, coord1], dim=1))

        in_ch = input_channels + 2  # +2 for CoordConv

        # 15x15 full-resolution stage
        self.conv_in = nn.Conv2d(in_ch, 32, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(32)
        self.res1 = _ResBlock(32)

        # Downsample to 8x8
        self.conv_down1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.res2 = _ResBlock(64)

        # Downsample to 4x4
        self.conv_down2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)

        flat_h = (d0 + 1) // 2   # 15 -> 8
        flat_h = (flat_h + 1) // 2  # 8 -> 4
        flat_w = (d1 + 1) // 2
        flat_w = (flat_w + 1) // 2
        flat_size = 64 * flat_h * flat_w

        if use_dueling:
            self.value_head = nn.Sequential(
                nn.Linear(flat_size, 256), nn.ReLU(), nn.Linear(256, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(flat_size, 256), nn.ReLU(), nn.Linear(256, n_actions),
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(flat_size, 256), nn.ReLU(), nn.Linear(256, n_actions),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)

        x = torch.cat([x, self._coords.expand(B, -1, -1, -1)], dim=1)

        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res1(x)
        x = F.relu(self.conv_down1(x))
        x = self.res2(x)
        x = F.relu(self.conv_down2(x))

        features = x.reshape(B, -1)

        if self.use_dueling:
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.q_head(features)


class DQN_CNN_Shallow(nn.Module):
    """
    Shallow CNN for grid-based observations on small grids (e.g. 15x15).

    Three plain conv layers with strided downsampling, no residual blocks:
      conv1 (5x5, 32ch, 15x15) -> conv2 (3x3 stride 2, 32ch, 8x8)
      -> conv3 (3x3 stride 2, 32ch, 4x4)
    Flatten (32*4*4 = 512) -> FC(128) -> Q-heads.

    Much fewer parameters than DQN_CNN; suited for dense distance-field
    observations where the input already carries strong spatial signal.
    """

    def __init__(
        self,
        input_channels: int = 6,
        grid_size: Tuple[int, int] = (15, 15),
        n_actions: int = 3,
        use_dueling: bool = True,
    ):
        super().__init__()
        self.use_dueling = use_dueling
        d0, d1 = grid_size

        coord0 = torch.linspace(-1, 1, d0).view(1, 1, d0, 1).expand(1, 1, d0, d1)
        coord1 = torch.linspace(-1, 1, d1).view(1, 1, 1, d1).expand(1, 1, d0, d1)
        self.register_buffer("_coords", torch.cat([coord0, coord1], dim=1))

        in_ch = input_channels + 2

        self.conv1 = nn.Conv2d(in_ch, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        flat_h = (d0 + 1) // 2
        flat_h = (flat_h + 1) // 2
        flat_w = (d1 + 1) // 2
        flat_w = (flat_w + 1) // 2
        flat_size = 32 * flat_h * flat_w

        if use_dueling:
            self.value_head = nn.Sequential(
                nn.Linear(flat_size, 128), nn.ReLU(), nn.Linear(128, 1),
            )
            self.advantage_head = nn.Sequential(
                nn.Linear(flat_size, 128), nn.ReLU(), nn.Linear(128, n_actions),
            )
        else:
            self.q_head = nn.Sequential(
                nn.Linear(flat_size, 128), nn.ReLU(), nn.Linear(128, n_actions),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = torch.cat([x, self._coords.expand(B, -1, -1, -1)], dim=1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        features = x.reshape(B, -1)

        if self.use_dueling:
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.q_head(features)
