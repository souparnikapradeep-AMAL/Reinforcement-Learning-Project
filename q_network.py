"""Q-Network architecture shared by DQN and Double DQN."""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Fully connected Q-network.
    Architecture: 2 hidden layers with 128 units each and ReLU activations.
    Input: state vector, Output: Q-value for each action.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
