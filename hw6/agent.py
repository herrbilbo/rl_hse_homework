import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import os

class MLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, out_size)

    def forward(self, state):
        res = F.relu(self.fc1(state))
        res = F.relu(self.fc2(res))
        return self.fc3(res)
        
def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_normal_(layer.weight)
        
        
class Exploration:
    def __init__(self, state_dim, action_dim, beta=1, lr=1e-3, exp_dim=8, target_dim=256, latent_space=16, eps=0.05):
        
        self.exploration = MLP(state_dim, latent_space, exp_dim)
        self.target = MLP(state_dim, latent_space, target_dim)
        
        
        self.exploration.apply(init_weights)
        self.optimizer = torch.optim.Adam(self.exploration.parameters(), lr=lr)
        self.beta = beta
        self.eps = eps

        self.target.apply(init_weights)
        self.target.eval()

    def get_exploration_reward(self, states, actions, next_states):
        target_pred = self.target(next_states)
        exp_pred = self.exploration(next_states)
        with torch.no_grad():
            return F.mse_loss(exp_pred, target_pred, reduction='none').sum(dim=1)

    def update(self, transition):
        state, action, next_state, reward, done = transition
        
        target_pred = self.target(torch.tensor(next_state).float()).detach()
        exp_pred = self.exploration(torch.tensor(next_state).float())
        
        loss = ((exp_pred - target_pred) ** 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
class Agent:
    def __init__(self, state_dim, action_dim, buffer_size=20000, batch_size=128, gamma=0.99, lr=1e-3, replace_target=50, actor_dim=256):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.replace_target = replace_target
        self.exploration = Exploration(self.state_dim, self.action_dim)
        
        self.local_model = MLP(self.state_dim, self.action_dim, actor_dim)
        self.local_model.apply(init_weights)
        self.target_model = MLP(self.state_dim, self.action_dim, actor_dim)
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=lr)
        
        self.memory = memory_buffer(self.buffer_size, self.batch_size)
        self.time = 0
    
    def update(self, transition):
        self.memory.add(transition)
        self.exploration.update(transition)
        self.time += 1
        if len(self.memory) > self.batch_size:
            batch = self.memory.sample()
            
            states = torch.from_numpy(np.vstack([e[0] for e in batch if e is not None])).float()
            actions = torch.from_numpy(np.vstack([e[1] for e in batch if e is not None])).long()
            next_states = torch.from_numpy(np.vstack([e[2] for e in batch if e is not None])).float()
            rewards = torch.from_numpy(np.vstack([e[3] for e in batch if e is not None])).float()
            dones = torch.from_numpy(np.vstack([e[4] for e in batch if e is not None]))
            
            #print(rewards)
            
            additional = (self.exploration.get_exploration_reward(states, actions, next_states)).reshape(self.batch_size, 1)
            
            additional = (additional - additional.mean()) / additional.std()
            
            #print(rewards.size(), additional.size())
            rewards = rewards + self.exploration.beta * additional
            #print(rewards.size())

            # DDQN
            with torch.no_grad():
                amax = self.local_model(next_states).detach().max(1)[1].unsqueeze(1)
                target_pred = rewards + ((~dones) * self.gamma * self.target_model(next_states).detach().gather(1, amax))
            # DDQN
                
            local_pred = self.local_model(states).gather(1, actions)

            #print(local_pred.size(), target_pred.size())
            loss = F.mse_loss(local_pred, target_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.time % self.replace_target == 0:
                self.target_model.load_state_dict(self.local_model.state_dict())  

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0)
            action_values = self.local_model(state)
            if random.random() < self.exploration.eps:
                return random.randint(0, self.action_dim - 1)
            return np.argmax(action_values.data.numpy())

    def save(self, path='agent.pkl'):
        torch.save(self.local_model.state_dict(), path)
        
    def reset(self):
        pass
     
     
class memory_buffer:
    def __init__(self, max_len, batch_size):
        self.mem = [None] * max_len
        self.max_len = max_len
        self.tail = 0
        self.len = 0
        self.batch_size = batch_size
    
    def add(self, transition):
        self.mem[self.tail] = transition
        self.len = min(self.len + 1, self.max_len)
        self.tail = (self.tail + 1) % self.max_len
    
    def sample(self):
        temp = random.sample(range(self.len), self.batch_size)
        return [self.mem[i] for i in temp]

    def __len__(self):
        return self.len