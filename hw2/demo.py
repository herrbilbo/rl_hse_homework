import gym
import random
import torch
import numpy as np
from collections import deque
from train import dqn
    
if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    #env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    
    agent = dqn(state_size=8, action_size=4)
    
    agent.local_model.load_state_dict(torch.load('gg.pkl'))

    for i in range(10):
        total = 0
        state = env.reset()
        hist = np.zeros(4)
        for j in range(1000):
            env.render()
            action = agent.act(state)
            hist[action] += 1
            state, reward, done, _ = env.step(action)
            total += reward
            if done:
                break
                
        print(total, hist)
    env.close()