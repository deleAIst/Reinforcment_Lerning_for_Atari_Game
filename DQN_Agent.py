import collections
import os
import random

from typing import Deque

import gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from DQN_Net import DQN
from DQN_Wapper import make_env

DQN_PATH=os.path.join("/home/dennis/Studium/RNFL_Atari/Reinforcment_Lerning_for_Atari_Game/Models")
TARGET_DQN_PATH=os.path.join("/home/dennis/Studium/RNFL_Atari/Reinforcment_Lerning_for_Atari_Game/Models")
class Agent:
    def __init__(self, game: str):
        # DQN Env Variables
        self.device = "cpu"
        self.num_buff_frames = 4
        self.env = make_env(game, self.num_buff_frames)
        self.img_shape = (84, 84, self.num_buff_frames)
        self.observations = self.env.observation_space.shape
        self.actions = self.env.action_space.n

        # DQN Agent Variables
        self.replay_buffer_size = 100_000
        self.train_start = 20_000
        self.memory: Deque = collections.deque(maxlen=self.replay_buffer_size)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_steps = 100_000
        self.epsilon_step = (
            self.epsilon - self.epsilon_min) / self.epsilon_steps
        # DQN Network Variables
        self.state_shape = self.observations
        self.learning_rate = 1e-1
        self.batch_size = 32
        self.dqn = DQN(84, 84, self.actions).to(self.device)
        self.target_dqn = DQN(84, 84, self.actions).to(self.device)
        self.target_dqn.load_state_dict(self.dqn.state_dict())
        self.target_dqn.eval()
        self.optimizer = optim.RMSprop(
            self.dqn.parameters(), lr=0.00025, eps=0.01)
        self.sync_models = 10_000

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.actions)
        else:
            return np.argmax(self.dqn(state))

    def train(self, num_episodes):
        last_rewards: Deque = collections.deque(maxlen=10)
        best_reward_mean = 0.0
        frame_it = 0
        for episode in range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            #state = np.reshape(state, newshape=(1, -1)).astype(np.float32)
            torch.tensor(data=state, dtype=torch.float32, device=self.device)
            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                #next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                next_state = torch.tensor(
                    data=next_state, dtype=torch.float32, device=self.device)
                self.remeber(state, action, reward, next_state, done)
                self.replay()
                total_reward += reward
                state = next_state
                if frame_it & self.sync_models == 0:
                    self.target_dqn.load_state_dict(self.dqn.state_dict())
                if done:
                    last_rewards.append(total_reward)
                    curremt_reward_mean = np.mean(last_rewards)
                    print(f"Episode: {episode} Reward: {total_reward} MeanReward: {round(curremt_reward_mean, 2)}"
                          f"Epsilon: {round(self.epsilon, 2)} MemSize: {len(self.memory)}")
                    if(curremt_reward_mean > best_reward_mean):
                        best_reward_mean = curremt_reward_mean
                        torch.save(self.dqn.state_dict(), DQN_PATH)
                        torch.save(self.target_dqn.state_dict(),
                                   TARGET_DQN_PATH)
                        print(f"New best mean: {best_reward_mean}")
                    break

    def episode_anneal(self):
        if self.memory < self.train_start:
            return
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_step

    def remeber(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = np.concatenate(states).astype(np.float32)
        #states = torch.cat(torch.Tensor(data=states, dtype=torch.float32, device=self.device ))
        next_states = np.concatenate(next_states).astype(np.float32)
        #next_states = torch.cat(torch.Tensor(data=next_states, dtype=torch.float32, device=self.device ))
        # actions = np.concatenate(actions)
        states = torch.from_numpy(states)
        next_states = torch.from_numpy(next_states)
        #actions = torch.Tensor(actions.astype(np.int32)).to(self.device)

        q_values = self.dqn(states)
        q_values_=q_values

        #q_values_next = torch.zeros(self.batch_size, device=self.device)
        q_values_next = self.target_dqn(next_states).max(1)[0].detach()
       # print(f"q_values: {q_values_next.size()} States: {states.size()}  q_values_: {q_values}")

        for i in range(self.batch_size):
            a = actions[i]
            done = dones[i]
            if done:
                q_values[i][a]= rewards[i]
            else:
                q_values[i][a] = rewards[i] + self.gamma * q_values_next[i]
        loss = F.smooth_l1_loss(q_values_, q_values )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def play(self, num_episodes, render=True):
        # TODO: Model Laden hier dqn und target dqn
        self.dqn.load_state_dict(torch.load(DQN_PATH))
        self.target_dqn.load_state_dict(torch.load(TARGET_DQN_PATH))

        for episode in self.range(1, num_episodes + 1):
            total_reward = 0.0
            state = self.env.reset()
            #state = np.reshape(state, newshape=(1, -1)).astype(np.float32)
            state = torch.tensor(
                data=state, dtype=torch.float32, device=self.device)
            while True:
                if render:
                    self.env.render()
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                #next_state = np.reshape(next_state, newshape=(1, -1)).astype(np.float32)
                next_state = torch.tensor(
                    data=next_state, dtype=torch.float32, device=self.device)
                total_reward += reward
                state = next_state
                if done:
                    print(
                        f"Episode: {episode} Reward: {total_reward} Epsilon: {self.epsilon}")


if __name__ == "__main__":
    game = "PongNoFrameskip-v4"
    agent = Agent(game)
    agent.train(num_episodes=3_000)
    input("Play?")
    agent.play(num_episodes=30, render=True)
