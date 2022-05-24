import gym
import collections
import random
import sys
import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#Hyperparameters
ENV_NAME = "CartPole-v1"

GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 100000
BATCH_SIZE = 100

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.996


class SmartBufferDQN(object):
    def __init__(self, state_dim=4, max_size=MEMORY_SIZE):     
        self.max_size = max_size
        self.state_dim = state_dim
        self.ptr = 0

        self.states = np.empty((max_size, state_dim))
        self.actions = np.empty((max_size, 1))
        self.next_states = np.empty((max_size, state_dim))
        self.rewards = np.empty((max_size, 1))
        self.dones = np.empty((max_size, 1))

    def add(self, s, a, s_p, r, d):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.next_states[self.ptr] = s_p
        self.rewards[self.ptr] = r
        self.dones[self.ptr] = d

        self.ptr += 1
        
        if self.ptr == 99999: self.ptr = 0

    def sample(self, batch_size):
        ind = np.random.randint(0, self.ptr-1, size=batch_size)
        states = np.empty((batch_size, self.state_dim))
        actions = np.empty((batch_size, 1))
        next_states = np.empty((batch_size, self.state_dim))
        rewards = np.empty((batch_size, 1))
        dones = np.empty((batch_size, 1))

        for i, j in enumerate(ind): 
            states[i] = self.states[j]
            actions[i] = self.actions[j]
            next_states[i] = self.next_states[j]
            rewards[i] = self.rewards[j]
            dones[i] = self.dones[j]

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.int64)
        next_states = torch.tensor(next_states, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)

        return states, actions, next_states, rewards, dones

    def size(self):
        return self.ptr


class Qnet(nn.Module):
    def __init__(self, obs_space, action_space, h_size):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(obs_space, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
      
class DQN:
    def __init__(self, obs_space, action_space, name="Agent"):
        self.obs_space = obs_space
        self.action_space = action_space
        self.replay_buffer = SmartBufferDQN(obs_space)

        self.name = name
        self.model = None 
        self.target = None
        self.optimizer = None

        self.exploration_rate = EXPLORATION_MAX
        self.update_steps = 0

        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)

    def create_agent(self, h_size):
        obs_space = self.obs_space
        action_space = self.action_space

        self.model = Qnet(obs_space, action_space, h_size)
        self.target = Qnet(obs_space, action_space, h_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def act(self, obs):
        if random.random() < self.exploration_rate:
            return random.randint(0, self.action_space-1)
        else: 
            return self.greedy_action(obs)

    def greedy_action(self, obs):
        obs_t = torch.from_numpy(obs).float()
        out = self.model.forward(obs_t)
        return out.argmax().item()

    def train(self):
        n_train = 2
        for i in range(n_train):
            if self.replay_buffer.size() < BATCH_SIZE:
                return
            s, a, s_p, r, done = self.replay_buffer.sample(BATCH_SIZE)

            next_values = self.target.forward(s_p)
            max_vals = torch.max(next_values, dim=1)[0].reshape((BATCH_SIZE, 1))
            g = torch.ones_like(done) * GAMMA
            q_update = r + g * max_vals * done
            q_vals = self.model.forward(s)
            q_a = q_vals.gather(1, a)
            loss = F.mse_loss(q_a, q_update.detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.update_networks()

    def update_networks(self):
        self.update_steps += 1
        if self.update_steps % 100 == 1: # every 20 eps or so
            self.target.load_state_dict(self.model.state_dict())
        if self.update_steps % 12 == 1:
            self.exploration_rate *= EXPLORATION_DECAY 
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

    def save(self, directory="./saves"):
        filename = self.name

        torch.save(self.model, '%s/%s_model.pth' % (directory, filename))
        torch.save(self.target, '%s/%s_target.pth' % (directory, filename))

    def load(self, directory="./saves"):
        filename = self.name

        self.model = torch.load('%s/%s_model.pth' % (directory, filename))
        self.target = torch.load('%s/%s_target.pth' % (directory, filename))

        print(f"Agent Loaded: {filename}")


class DQN_test:
    def __init__(self, obs_space, action_space, name="Agent"):
        self.obs_space = obs_space
        self.action_space = action_space

        self.name = name
        self.model = None 

        self.update_steps = 0

        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)

        self.load()

    def act(self, obs):
        return self.greedy_action(obs)

    def greedy_action(self, obs):
        obs_t = torch.from_numpy(obs).float()
        out = self.model.forward(obs_t)
        return out.argmax().item()

    def load(self, directory="./saves"):
        filename = self.name

        self.model = torch.load('%s/%s_model.pth' % (directory, filename))
        self.target = torch.load('%s/%s_target.pth' % (directory, filename))

        print(f"Agent Loaded: {filename}")




def observe(env, memory, n_itterations=10000):
    s = env.reset()
    done = False
    for i in range(n_itterations):
        action = env.action_space.sample()
        s_p, r, done, _ = env.step(action)
        done_mask = 0.0 if done else 1.0
        memory.add(s, action, s_p, r/100, done_mask)
        # memory.put((s, action, r/100, s_p, done_mask))
        s = s_p
        if done:
            s = env.reset()

        print("\rPopulating Buffer {}/{}.".format(i, n_itterations), end="")
        sys.stdout.flush()

def test_cartpole():
    env = gym.make('CartPole-v1')
    dqn = DQN(env.observation_space.shape[0], env.action_space.n, "AgentCartpole")
    dqn.create_agent(100)

    print_n = 20

    rewards = []
    observe(env, dqn.replay_buffer)
    for n in range(500):
        score, done, state = 0, False, env.reset()
        while not done:
            a = dqn.act(state)
            s_prime, r, done, _ = env.step(a)
            done_mask = 0.0 if done else 1.0
            dqn.replay_buffer.add(state, a, s_prime, r/100, done_mask)
            # dqn.memory.put((state, a, r/100, s_prime, done_mask))
            state = s_prime
            score += r
            dqn.train()

        dqn.save()
            
        rewards.append(score)
        if n % print_n == 1:
            print(f"Run: {n} --> Score: {score} --> Mean: {np.mean(rewards[-20:])} --> exp: {dqn.exploration_rate}")

def test_():
    env = gym.make('CartPole-v1')
    dqn = DQN_test(env.observation_space.shape[0], env.action_space.n, "AgentCartpole")


    rewards = []
    for n in range(5):
        score, done, state = 0, False, env.reset()
        while not done:
            a = dqn.act(state)
            state, r, done, _ = env.step(a)
            score += r

        rewards.append(score)
        # if n % print_n == 1:
        print(f"Run: {n} --> Score: {score} --> Mean: {np.mean(rewards[-20:])}")


if __name__ == '__main__':
    # test_cartpole()
    test_()

