import collections
import os
import random
import datetime

from pathlib import Path
from collections import deque

import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT



from DQN_Net import DQN
from DQN_Wapper import make_env
from Logging import MetricLogger

DQN_PATH=os.path.join("/home/dennis/Studium/RNFL_Atari/Reinforcment_Lerning_for_Atari_Game/Models")
TARGET_DQN_PATH=os.path.join("/home/dennis/Studium/RNFL_Atari/Reinforcment_Lerning_for_Atari_Game/Models")
class Agent:
    def __init__(self, state_dim, action_dim, save_dir, env):
        
        self.env = env
        # DQN Env Variables
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        
        self.use_cuda = torch.cuda.is_available()
        
        self.net = DQN(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device="cuda")
            
        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0
        
        self.save_every= 5e5
        
        
        self.memory = deque(maxlen=100000)
        self.batch_size = 32
        
        self.gamma = 0.9
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()
        
        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4 

    def get_action(self, state):
        """
    Given a state, choose an epsilon-greedy action and update value of step.

    Inputs:
    state(LazyFrame): A single observation of the current state, dimension is (state_dim)
    Outputs:
    action_idx (int): An integer representing which action Mario will perform
        """
        #Explore
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        
        #Exploit
        else:
            state = state.__array__()
            if self.use_cuda:
                state = torch.tensor(state).cuda()
            else:
                state = torch.tensor(state)
            state = state.unsqueeze(0)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()
            
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        
        self.curr_step +=1
        return action_idx
    
    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = state.__array__()
        next_state = next_state.__array__()
        
        if self.use_cuda:
            state = torch.tensor(state).cuda()
            next_state = torch.tensor(next_state).cuda()
            action = torch.tensor([action]).cuda()
            reward = torch.tensor([reward]).cuda()
            done = torch.tensor([done]).cuda()
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))
        
    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
    
    
    def td_estimator(self, state, action):
        current_Q = self.net(state, model="online")[np.arange(0, self.batch_size), action]#Q(s,a)
        return current_Q
    
    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action]
        return (reward + (1- done.float())* self.gamma *next_Q).float()
    
    def update_Q_online(self, td_estimator, td_target):
        loss = self.loss_fn(td_estimator, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())
        
    def save(self):
        save_path = (
            self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")
        
        
    def lerne(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
            
        if self.curr_step % self.save_every == 0:
            self.save()
            
        if self.curr_step < self.burnin:
            return None, None
        
        if self.curr_step % self.learn_every != 0:
            return None, None
        
        state, next_state, action, reward, done = self.recall()
        
        td_est = self.td_estimator(state, action)
        
        td_tgt = self.td_target(reward, next_state, done)
        
        loss = self.update_Q_online(td_est, td_tgt)
        
        return (td_est.mean().item(), loss)
    
    def run(self, save_dir, episodes):
        logger = MetricLogger(save_dir)
        for e in range(episodes):
            
            state = self.env.reset()
            
            while True:
                #self.env.render()
                action = self.get_action(state)
                
                next_state, reward, done, info = self.env.step(action)
                
                self.cache(state, next_state, action, reward, done)
                
                q, loss = self.lerne()
                
                logger.log_step(reward, loss, q)
                
                state = next_state
                
                if done:
                    break
                
            logger.log_episode()
            
            if e % 20 == 0:
                logger.record(episode=e, epsilon=self.exploration_rate, step=self.curr_step)
        


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    print()

    save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True)
    
    
    game = "SuperMarioBros-1-1-v0"
    env =make_env(game, 4)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    agent = Agent(state_dim=(4, 84, 84), action_dim=env.action_space.n, save_dir=save_dir, env=env)
    agent.run(save_dir, 10)
    #input("Play?")
    #agent.play(num_episodes=30, render=True)
