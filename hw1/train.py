# https://courses.cs.washington.edu/courses/cse473/16au/slides-16au/18-approx-rl2.pdf

from gym import make
import numpy as np
#import torch
#import copy
from collections import deque
import random


def transform_state(state):
    x, v = state
    x = (x + 1.2) / 1.8
    v = (v + 0.07) / 0.14
    return np.array([x, v]).reshape(2, 1)

class AQL:
    def __init__(self, state_dim, action_dim, lr, gamma):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma

        self.weight = np.random.normal(loc=0, scale=0.01, size=(self.action_dim, self.state_dim))
        self.bias = np.random.normal(loc=0, scale=0, size=(self.action_dim, 1))
        
    def _get_prediction(self, state):
        return (self.weight @ transform_state(state) + self.bias)

    def update(self, transition):
        state, action, next_state, reward, done = transition
        
        difference = reward + self.gamma * np.max(self._get_prediction(next_state)) - (self._get_prediction(state))[action]
        
        self.weight[action] = self.weight[action] + self.lr * difference * transform_state(state).T
        self.bias[action] = self.bias[action] + self.lr * difference

    def act(self, state, target=False):
        return np.argmax(self._get_prediction(state))

    def save(self, path):
        weight = np.array(self.weight)
        bias = np.array(self.bias)
        np.savez(path, weight, bias)
        
    def check_agent(self):
        local_env = make("MountainCar-v0")
        res = []
        for _ in range(20):
            state = local_env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = local_env.step(action)
                total_reward += reward
                state = next_state
            res.append(total_reward)
        return res


if __name__ == "__main__":

    env = make("MountainCar-v0")
    seed = 228
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    aql = AQL(state_dim=2, action_dim=3, lr=0.05, gamma=0.5)
    history = []
    best = -200
    eps = 1
    episodes = 2000
    n_step = 3
    last_save = -1
    
    for i in range(episodes):
        state = env.reset().reshape(aql.state_dim, 1)
        total_reward = 0
        steps = 0
        done = False
        eps *= 0.997
        
        reward_buffer = deque(maxlen=n_step)
        state_buffer = deque(maxlen=n_step)
        action_buffer = deque(maxlen=n_step)
        
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = aql.act(state)
                
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            reward = reward + 300 * abs(next_state[1])
            
            next_state = next_state.reshape(aql.state_dim, 1)
            
            
            steps += 1
            reward_buffer.append(reward)
            state_buffer.append(state)
            action_buffer.append(action)
            
            if len(reward_buffer) == n_step:
                aql.update((state_buffer[0], action_buffer[0], next_state, sum([(aql.gamma ** i) * r for i, r in enumerate(reward_buffer)]), done))
            state = next_state
            
        history.append(total_reward)
        
        if len(reward_buffer) == n_step:
            rb = list(reward_buffer)
            for k in range(1, n_step):
                aql.update((state_buffer[k], action_buffer[k], next_state, sum([(aql.gamma ** i) * r for i, r in enumerate(rb[k:])]), done))

        if i % 20 == 0 and i > 0:
            print(f"I've done episodes {i-20}-{i}, cur eps is {eps}, the last save was on episode {last_save}")
            print(f'Over these 20 episodes:')
            print(f'best reward: {np.max(history)}, worst reward: {np.min(history)}, mean reward: {np.mean(history)}')
            print('-' * 100)
            history = []
            
            cur = np.mean(aql.check_agent())
            if cur > best:
                last_save = i
                best = cur
                print(f"I found better model, it's score is {best}, resaving it as 'agent.npz'")
                print('-' * 100)
                aql.save('agent.npz')