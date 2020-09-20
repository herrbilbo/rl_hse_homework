import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import random

class MLP(nn.Module):
    def __init__(self, state_size=8, action_size=4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
        
        
class Agent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = MLP()
        self.model.load_state_dict(torch.load(__file__[:-8] + "/gg.pkl", map_location=self.device))
        self.model.eval()
        
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.model(state)
        return np.argmax(action.cpu().data.numpy())

    def reset(self):
        pass