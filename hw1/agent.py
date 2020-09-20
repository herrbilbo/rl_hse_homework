import random
import numpy as np
import os
#from train import transform_state

def transform_state(state):
    x, v = state
    x = (x + 1.2) / 1.8
    v = (v + 0.07) / 0.14
    return np.array([x, v]).reshape(2, 1)


class Agent:
    def __init__(self):
        agent = np.load(__file__[:-8] + "/agent.npz")
        self.weight = agent[agent.files[0]]
        self.bias = agent[agent.files[1]]

    def act(self, state, target=False):
        return np.argmax(self.weight @ transform_state(state) + self.bias)

    def reset(self):
        pass
