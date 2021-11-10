# -*- coding: utf-8 -*-
"""
tutorial source : https://www.youtube.com/watch?v=G0L8SN02clA
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
from gym import wrappers

class GenericNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(GenericNetwork,self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        
        self.optimizer= optim.Adam(self.parameters(), lr = self.lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        
    def forward(self, observation):
        state = T.tensor(observation, dtype = T.float).to(self.device)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Agent(object):
    def __init__(self, alpha, beta, input_dims, gamma = 0.99, n_actions = 2, layer1_size = 64, layer2_size = 64, n_outputs = 1):
        self.gamma = gamma
        self.log_probs = None
        self.n_outputs = n_outputs
        self.actor = GenericNetwork(alpha, input_dims, layer1_size, layer2_size, n_actions = n_actions)
        self.critic = GenericNetwork(beta, input_dims, layer1_size, layer2_size, n_actions = 1)
        
    def choose_action(self, observation):
        mu, sigma = self.actor.forward(observation)
        sigma = T.exp(sigma)
        action_probs = T.distributions.Normal(mu, sigma)
        probs = action_probs.sample(sample_shape = T.Size([self.n_outputs]))
        self.log_probs = action_probs.log_prob(probs).to(self.actor.device)
        action = T.tanh(probs)   ##depends on limits of action domain
        return action.item()
    
    def learn(self, state, reward, new_state, done):
        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()
        
        critic_value_ = self.critic.forward(new_state)
        critic_value = self.critic.forward(state)
        
        reward = T.tensor(reward, dtype=T.float).to(self.actor.device)
        delta = reward + self.gamma * critic_value_ * (1 - int(done)) - critic_value
        
        actor_loss = -self.log_probs * delta
        critic_loss = delta ** 2
        
        (actor_loss + critic_loss).backward()
        self.actor.optimizer.step()
        self.critic.optimizer.step()
        

        
        
        
if __name__ == '__main__':
    agent = Agent(alpha=0.000005, beta=0.00001, input_dims=[2], gamma=0.99, layer1_size=256, layer2_size=256)
    env = gym.make('MountainCarContinuous-v0')
    score_history = []
    num_episodes = 100
    for i in range(num_episodes):
        done = False
        score = 0
        reward_history = []
        action_history = []
        velocity_history = []
        success = 0
        observation = env.reset()
        env.render()
        while not done:
            action = np.array(agent.choose_action(observation)).reshape((1,))
            observation_, reward, done, info = env.step(action)
            env.render()
            if info != "{}":
                reward =  observation[1] * action.item() / 0.07 - 0.1 ##hoping to get "correct" solution
            else:
                reward =  observation[1] * action.item() / 0.07 - 1  + int(done) * 100
                if done:
                    success += 1
            action_history.append(action.item())
            velocity_history.append(observation[1]/0.07)
            reward_history.append(reward)
            agent.learn(observation, reward, observation_, done)
            observation = observation_
            score += reward
        plt.plot(action_history, 'red', linestyle = '', marker = 'o')
        plt.plot(velocity_history, 'green')
        plt.plot(reward_history, 'blue')
        plt.show()
        plt.figure
        plt.hist(action_history)
        plt.show()
        score_history.append(score)
        print(info)
        print('episode ', i, ': score %.2f' % score)
    plt.figure
    plt.plot(score_history)
    plt.show()
    print(success)
    env.close()