from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

import torch
from torch import nn


class TerrainCNNExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 160):
        super().__init__(observation_space, features_dim)

        # CNN for terrain: (1, 32, 32) -> 32
        self.terrain_cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),   # -> (8, 16, 16)
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # -> (16, 8, 8)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # -> (32, 4, 4)
            nn.ReLU(),
            nn.Flatten(),                                           # -> 512
            nn.Linear(512, 32),
            nn.ReLU(),
        )

        self.robot_mlp = nn.Sequential(
            nn.Linear(observation_space["robot_state"].shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 12833),
            nn.ReLU(),
        )

    def forward(self, observations):
        terrain_features = self.terrain_cnn(observations["terrain"])
        robot_features = self.robot_mlp(observations["robot_state"])
        return torch.cat([robot_features, terrain_features], dim=1)