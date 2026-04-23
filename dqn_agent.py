"""DQN / Double DQN Agent.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from q_network import QNetwork
from replay_buffer import ReplayBuffer


class DQNAgent:
    """Agent that learns via DQN or Double DQN.

    Args:
        state_dim: Dimensionality of the observation space.
        action_dim: Number of discrete actions.
        use_double: If True, use Double DQN target; otherwise standard DQN.
        lr: Learning rate for the Adam optimizer.
        gamma: Discount factor.
        buffer_size: Maximum replay buffer capacity.
        batch_size: Mini-batch size for training.
        target_update_freq: Copy online -> target network every N steps.
        epsilon_start: Initial exploration rate.
        epsilon_end: Final exploration rate.
        epsilon_decay_steps: Number of steps to linearly decay epsilon.
        device: Torch device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        use_double: bool = False,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_size: int = 50_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay_steps: int = 10_000,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.use_double = use_double
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Epsilon schedule
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps

        # Networks
        self.online_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()  

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        # Replay buffer
        self.buffer = ReplayBuffer(capacity=buffer_size)

        # Step counter 
        self.step_count = 0

  
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Epsilon-greedy action selection.

        In eval_mode, uses a fixed small epsilon (0.05) instead of the
        decaying training epsilon.
        """
        eps = 0.05 if eval_mode else self.epsilon
        if np.random.random() < eps:
            return np.random.randint(self.action_dim)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return q_values.argmax(dim=1).item()


    # Training step
 
    def train_step(self) -> dict:
        """Sample a mini-batch and perform one gradient update.

        Returns a dict with training metrics (loss, mean Q-value).
        """
        if len(self.buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Current Q-values for chosen actions
        current_q = self.online_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

    
        with torch.no_grad():
            if self.use_double:
                best_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states_t).gather(1, best_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states_t).max(dim=1)[0]

            targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = self.loss_fn(current_q, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update step counter, epsilon, and target network
        self.step_count += 1
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return {
            "loss": loss.item(),
            "mean_q": current_q.mean().item(),
        }

    # Convenience: get Q-values for a batch of states
    def estimate_q_values(self, states: np.ndarray) -> float:
        """Return the mean max-Q across a batch of states (for tracking overestimation)."""
        states_t = torch.FloatTensor(states).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(states_t)
        return q_values.max(dim=1)[0].mean().item()


    def save(self, path: str):
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.step_count = checkpoint["step_count"]
        self.epsilon = checkpoint["epsilon"]
