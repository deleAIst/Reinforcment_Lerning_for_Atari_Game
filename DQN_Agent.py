import collections
import os
import random

from typing import Deque

import gym
import numpy as np

from DQN_Net import DQN
from DQN_WAPPER import make_env


class Agent:
    def __init__(self, env: gym.Env):
        # DQN Env Variables
        self.env = env
        self.observations = self.env.observation_space.shape
        self.action = self.env.action_space.n
        # DQN Agent Variables
        self.replay_buffer_size = 50_000
        self.train_start = 1_000
        self.memory: Deque = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # DQN Network Variables
        self.state_shape = self.observations
        self.learning_rate = 1e-1
