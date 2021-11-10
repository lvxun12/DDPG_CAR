##Imports
from collections import deque
import random
import copy
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt
from gym import wrappers
from torch.optim import Adam


##Hyperparameters
BUFFER_SIZE = 100
BATCH_SIZE=64
GAMMA=0.99
TAU=0.001       #Target Network HyperParameters Update rate
LRA=0.0001      #LEARNING RATE ACTOR
LRC=0.001       #LEARNING RATE CRITIC
H1=400   #neurons of 1st layers
H2=300   #neurons of 2nd layers
INIT_W = 0.003  #initial weights' limit
MAX_EPISODES=5000 #number of episodes of the training
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995
EXPLORATION_SIGMA = 0.2
ENV_NAME = 'MountainCarContinuous-v0'
device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
default_seed = 0
RENDER = True
EVALUATION_EPISODES = 10

# Soft update of target networks
def soft_sync(src, dst, tau=TAU):
    with T.no_grad():
        for p, p_targ in zip(src.parameters(), dst.parameters()):
            p_targ.data.mul_(1 - tau)
            p_targ.data.add_(tau * p.data)

##Noise
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mean, sigma, theta=.15, time=1e-2, init_x=None):
        self.theta = theta
        self.mean = mean
        self.sigma = sigma
        self.time = time
        self.init_x = init_x
        self.prev_x = None
        self.reset()

    def __call__(self):
        normal = np.random.normal(size=self.mean.shape)
        new_x = self.prev_x + self.theta * (self.mean - self.prev_x) \
            * self.time + self.sigma * np.sqrt(self.time) * normal
        self.prev_x = new_x
        return new_x

    def reset(self):
        if self.init_x is not None:
            self.prev_x = self.init_x
        else:
            self.prev_x = np.zeros_like(self.mean)

##Replay Buffer
class ReplayBuffer(object):
    def __init__(self, buffer_size = BUFFER_SIZE, name_buffer=''):
        self.buffer_size=buffer_size
        self.num_exp=0
        self.buffer=deque()

    def add(self, s, a, r, s2, t):
        experience=(s, a, r, s2, t)
        if self.num_exp < self.buffer_size:
            self.buffer.append(experience)
            self.num_exp +=1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.buffer_size

    def count(self):
        return self.num_exp

    def sample(self, batch_size = BATCH_SIZE):
        if self.num_exp < batch_size:
            batch=random.sample(self.buffer, self.num_exp)
        else:
            batch=random.sample(self.buffer, batch_size)
        s, a, r, s2, t = map(np.stack, zip(*batch))
        return s, a, r, s2, t

    def clear(self):
        self.buffer = deque()
        self.num_exp=0


##Normalized Environment
class NormalizedEnv(gym.ActionWrapper):
    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)
    
    
##Networks
def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return T.Tensor(size).uniform_(-v, v)

class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=H1, hidden2=H2, init_w=INIT_W):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, nb_actions)
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, state):
        h = self.fc1(state)
        h = T.tanh(h)
        h = self.fc2(h)
        h = T.tanh(h)
        h = self.fc3(h)
        return T.tanh(h)

class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden1=H1, hidden2=H2, init_w=INIT_W):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden1)
        self.fc2 = nn.Linear(hidden1+nb_actions, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.init_weights(init_w)
    
    def init_weights(self, init_w):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
    
    def forward(self, state, action):
        h = self.fc1(state)
        h = T.tanh(h)
        h = self.fc2(T.cat([h,action], dim = 1))
        h = T.tanh(h)
        h = self.fc3(h)
        return h
    

##DDPG
env = NormalizedEnv(gym.make(ENV_NAME))
    
NB_states = env.observation_space.shape[0]
NB_actions = env.action_space.shape[0]
    
class DDPG(object):
    def __init__(self, nb_states = NB_states, nb_actions = NB_actions):
        super().__init__()
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.critic  = Critic(self.nb_states, self.nb_actions)
        self.actor = Actor(self.nb_states, self.nb_actions)
        self.target_critic = copy.deepcopy(self.critic)
        self.target_actor = copy.deepcopy(self.actor)
        self.critic_optimizer  = Adam(self.critic.parameters(), lr=LRC)
        self.actor_optimizer  = Adam(self.actor.parameters(), lr=LRA)
        self.memory = ReplayBuffer()
        self.exploration = EXPLORATION_MAX
        
    def forward(self, state):
        return self.actor(state)

    def evaluate(self, state):
        act = self.actor(T.tensor([state], dtype=T.float32, device='cpu:0'))
        return act.cpu().detach().numpy()[0]
    
    def compute_actor_loss(self, state):
        act = self.actor(state)
        value = self.critic(state, act)
        return -value.mean()

    def compute_critic_loss(self, state, action, reward, next_state, terminal):
        with T.no_grad():
            next_action = self.target_actor(next_state)
            next_value = self.target_critic(next_state, next_action)
            target = reward + self.gamma * next_value * (1.0 - terminal)

        value = self.critic(state, action)

        return ((value - target) ** 2).mean()

    def update_target(self):
        soft_sync(self.actor, self.target_actor)
        soft_sync(self.critic, self.target_critic)

def update(model, buffer, actor_optimizer, critic_optimizer, batch_size = BATCH_SIZE):
    batch = random.sample(range(buffer), batch_size)
    states = []
    actions = []
    rewards = []
    next_states = []
    terminals = []
    for state, action, reward, next_state, terminal in batch:
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_state)
        terminals.append(terminal)

    # numpy to torch.Tensor
    states = T.tensor(states, dtype=T.float32, device='cpu:0')
    actions = T.tensor(actions, dtype=T.float32, device='cpu:0')
    rewards = T.tensor(rewards, dtype=T.float32, device='cpu:0')
    next_states = T.tensor(next_states, dtype=T.float32, device='cpu:0')
    terminals = T.tensor(terminals, dtype=T.float32, device='cpu:0')

    # update critic
    critic_loss = model.compute_critic_loss(states, actions, rewards, next_states, terminals)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # update actor
    actor_loss = model.compute_actor_loss(states)
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    # update target networks
    model.update_target()
    
def evaluate(env, model, num_episodes, render):
    episode_rewards = []
    for _ in range(num_episodes):
        episode_reward = 0.0
        state = env.reset()
        while True:
            action = model.evaluate(state)
            state, reward, terminal, _ = env.step(action * env.action_space.high)
            episode_reward += reward;

            if render:
                env.render()

            if terminal:
                break
        episode_rewards.append(episode_reward)
    return np.mean(episode_rewards)


def main():

    env = gym.make(ENV_NAME)
    eval_env = gym.make(ENV_NAME)
    NB_states = env.observation_space.shape[0]
    NB_actions = env.action_space.shape[0]

    # seed everything
    env.seed(default_seed)
    eval_env.seed(default_seed)
    np.random.seed(default_seed)
    random.seed(default_seed)
    T.manual_seed(default_seed)
    T.cuda.manual_seed(default_seed)
    T.backends.cudnn.deterministic = True

    # ddpg model
    model = DDPG()

    # optimizers
    actor_optimizer = Adam(model.actor.parameters(), lr=LRA)
    critic_optimizer = Adam(model.critic.parameters(), lr=LRC)

    # replay buffer
    buffer = ReplayBuffer()

    # exploration noise
    noise = OrnsteinUhlenbeckActionNoise(
        np.zeros(NB_actions), np.ones(NB_actions) * EXPLORATION_SIGMA)

    t = 0
    while t < MAX_EPISODES:
        print(t)
        state = env.reset()
        terminal = False
        episode_reward = 0.0
        while not terminal:
            # inference
            action = model.evaluate(state)

            # add exploration noise
            action = np.clip(action + noise(), -1.0, 1.0)

            # interact with environment
            next_state, reward, terminal, _ = env.step(action * env.action_space.high)

            # store transition
            buffer.add(state, action, [reward], next_state, [terminal])

            # update parameters
            if buffer.num_exp > BATCH_SIZE:
                update(model, buffer, BATCH_SIZE, actor_optimizer,
                       critic_optimizer)

            if t % 1000 == 0:
                eval_reward = evaluate(eval_env, model, EVALUATION_EPISODES, RENDER)
                print('evaluate', t, eval_reward)

            state = next_state
            episode_reward += reward
            t += 1
        print(t, episode_reward)
    
if __name__ == '__main__':
    main()