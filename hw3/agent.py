import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class actor(nn.Module):
    def __init__(self, state_dim):
        super(actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = F.tanh(self.fc3(x))
        sigma = F.softplus(self.fc4(x))
        return 2 * mu, sigma + 1e-4


class Agent:
    def __init__(self):
        self.model = actor(3)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/gg.pickle"))

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            mu, sigma = self.model(state)
        norm = Normal(mu, sigma)
        action = torch.clamp(norm.sample().view(1), min=-2.0, max=2.0)
        return np.array([action.item()])

    def reset(self):
        pass
