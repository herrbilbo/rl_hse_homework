import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(actor, self).__init__()
        self.linear1 = nn.Linear(state_size, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, action_size)

    def forward(self, state):
        res = F.relu(self.linear1(state))
        res = F.relu(self.linear2(res))
        res = torch.tanh(self.linear3(res))
        return res

        
class Agent:
    def __init__(self):
        self.model = actor(28, 8)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/gg.pkl"))
        
    def act(self, state):
        state = torch.tensor(state).float()
        with torch.no_grad():
            action = self.model(state).cpu().data.numpy()
            
        noise = np.random.normal(loc=0.0, scale=0.03, size=action.shape)
        action = action + noise
        action = np.clip(action, -1.0, 1.0)
        return action

    def reset(self):
        pass