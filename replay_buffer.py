"""Experience Replay Buffer for DQN and Double DQN."""

import random
from collections import deque, namedtuple

import numpy as np

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity: int = 50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a random mini-batch of transitions.

        Returns numpy arrays for efficient conversion to tensors.
        """
        transitions = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))

        states = np.array(batch.state, dtype=np.float32)
        actions = np.array(batch.action, dtype=np.int64)
        rewards = np.array(batch.reward, dtype=np.float32)
        next_states = np.array(batch.next_state, dtype=np.float32)
        dones = np.array(batch.done, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
