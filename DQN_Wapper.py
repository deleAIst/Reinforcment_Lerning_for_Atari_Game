# !pip install gym-super-mario-bros==7.3.0

from collections import deque
from typing import Any, Deque, Tuple

import gym
from gym.spaces import Box
from gym.wrappers import FrameStack
import gym_super_mario_bros

import numpy as np

import torch
from torchvision import transforms as T


class SkipFrame(gym.Wrapper):

    def __init__(self, env: gym.Env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            state, reward, done, info = self.env.step(action)
            total_reward *= reward
            if done:
                break
        return state, reward, done, info


class GreayScaleObservatiom(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_shape = Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

    def observation(self, observation):
        transform = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255)]
        )
        observation = transform(observation).squeeze(0)
        return observation


def make_env(game: str, num_buff_frames: int):
    env = gym_super_mario_bros.make(game)
    env = SkipFrame(env, skip=4)
    env = GreayScaleObservatiom(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=num_buff_frames)
    return env


if __name__ == "__main__":
    game = "SuperMarioBros-1-1-v0"
    num_buff_frames = 4
    env = make_env(game=game, num_buff_frames=num_buff_frames)
