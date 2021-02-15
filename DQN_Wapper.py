import collections
from typing import Any, Deque, Tuple

import gym
import numpy as np

class StartGameWapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        env.reset()
        
    def reset(self, **kwargs: Any):
        self.env.reset()
        observation, _, _, _ = self.env.step(1) #FIRE
        return observation
    

class FrameStartGameWapper(gym.Wrapper):
    def __init__(self, env: gym.Env, num_buff_frames: int):
        super().__init__(env)
        self.num_buff_frames = num_buff_frames
        self.frames: Deque = collections.deque(maxlen=self.num_buff_frames)
        low = np.repeat(self.observation_space.low[np.newaxis, ], repeats=self.num_buff_frames,axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ], repeats=self.num_buff_frames,axis=0)
        self.observation_space= gym.spaces.Box(low=low, high=high, dtype=self.observation_space.dtype)
        
    def step(self, action: int)-> Tuple[np.ndarray, float, bool, dict]:
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        frame_stack = np.asarray(self.frames, dtype=np.float32) #(4, 84, 84)
        frame_stack = np.moveaxis(frame_stack, 0, 0)#(84, 84, 4)
        frame_stack = np.expand_dims(frame_stack, 0) #(1, 84, 84, 4)
        return frame_stack, reward, done, info
    
    def reset(self, **kwargs: Any) -> np.ndarray:
        self.env.reset(**kwargs)
        self.frames: Deque = collections.deque(maxlen=self.num_buff_frames)
        for _ in range(self.num_buff_frames):
            self.frames.append(np.zeros(shape=(84,84), dtype=np.float32))
        frame_stack = np.zeros(shape=(1, 84, 84, 4), dtype=np.float32)
        return frame_stack
    
def make_env(game: str, num_buff_frames: int):
    env = gym.make(game)
    env = gym.wrappers.AtariPreprocessing(env=env, 
                                          noop_max=20, 
                                          frame_skip=4, 
                                          screen_size=84, 
                                          terminal_on_life_loss=False, 
                                          grayscale_obs=True, 
                                          scale_obs=True)
    
    env = FrameStartGameWapper(env=env, num_buff_frames=num_buff_frames)
    env =StartGameWapper(env=env)
    return env


if __name__ == "__main__":
    game = "PongNoFrameskip-v4"
    num_buff_frames = 4
    env = make_env(game=game, num_buff_frames=num_buff_frames)
    
