"""Experience replay buffer for QMIX."""
import random, numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, actions, reward, next_obs, done, state, next_state):
        self.buffer.append((obs, actions, reward, next_obs, done, state, next_state))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones, states, next_states = zip(*batch)
        return (np.array(obs), np.array(actions), np.array(rewards, dtype=np.float32),
                np.array(next_obs), np.array(dones, dtype=np.float32),
                np.array(states), np.array(next_states))

    def __len__(self):
        return len(self.buffer)
