import gym
import random
import numpy as np
from collections import deque
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self, state_size, action_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        res = F.relu(self.fc1(state))
        res = F.relu(self.fc2(res))
        return self.fc3(res)

        
class dqn():
    def __init__(self, state_size, action_size, buffer_size=100000, batch_size=64, gamma=0.99, lr=5e-4, replace_target=100):
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.replace_target = replace_target
        
        self.local_model = MLP(state_size, action_size).to(device)
        self.target_model = MLP(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=lr)
        
        self.memory = memory_buffer(self.buffer_size, self.batch_size)
        self.time = 0
    
    def update(self, transition):
        state, action, reward, next_state, done = transition
        self.memory.add(state, action, reward, next_state, done)
        self.time += 1
        if len(self.memory) > self.batch_size:
            batch = self.memory.sample()
            
            states = torch.from_numpy(np.vstack([e[0] for e in batch if e is not None])).float().to(device)
            actions = torch.from_numpy(np.vstack([e[1] for e in batch if e is not None])).long().to(device)
            rewards = torch.from_numpy(np.vstack([e[2] for e in batch if e is not None])).float().to(device)
            next_states = torch.from_numpy(np.vstack([e[3] for e in batch if e is not None])).float().to(device)
            dones = torch.from_numpy(np.vstack([e[4] for e in batch if e is not None])).to(device)

            #with torch.no_grad():
            #    target_pred = rewards + ((~dones) * self.gamma * self.target_model(next_states).detach().max(1)[0].unsqueeze(1))
            
            
            # DDQN
            with torch.no_grad():
                amax = self.local_model(next_states).detach().max(1)[1].unsqueeze(1)
                target_pred = rewards + ((~dones) * self.gamma * self.target_model(next_states).detach().gather(1, amax))
            # DDQN
                
            local_pred = self.local_model(states).gather(1, actions)

            loss = F.mse_loss(local_pred, target_pred)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.time % self.replace_target == 0:
                self.target_model.load_state_dict(self.local_model.state_dict())  

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = self.local_model(state)
        return np.argmax(action_values.cpu().data.numpy())

    def save(self, path='agent.pkl'):
        torch.save(self.local_model.state_dict(), path)
     
     
class memory_buffer:
    def __init__(self, max_len, batch_size):
        self.mem = [None] * max_len
        self.max_len = max_len
        self.tail = 0
        self.len = 0
        self.batch_size = batch_size
    
    def add(self, state, action, reward, next_state, done):
        self.mem[self.tail] = (state, action, reward, next_state, done)
        self.len = min(self.len + 1, self.max_len)
        self.tail = (self.tail + 1) % self.max_len
    
    def sample(self):
        temp = random.sample(range(self.len), self.batch_size)
        return [self.mem[i] for i in temp]

    def __len__(self):
        return self.len

        
if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    print(device)
    seed = 228
    
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    episodes = 2000
    eps_border = 1e-7
    decay_rate = 0.997
    eps = 1.0
    
    state_size = 8
    action_size = 4
    
    agent = dqn(state_size=state_size, action_size=action_size)
    history = deque(maxlen=100)
    
    for episode in range(episodes):
        state = env.reset()
        score = 0
        eps = max(eps_border, eps * decay_rate)
        done = False
        while not done:
            if random.random() < eps:
                action = random.choice(np.arange(action_size))
            else:
                action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            transition = state, action, reward, next_state, done
            agent.update(transition)
            state = next_state
            score += reward  
        history.append(score)
        print(f'\repisode {episode}, mean: {np.mean(history)}, variance: {np.sqrt(np.var(history))}, min: {np.min(history)}, max: {np.max(history)}, eps: {eps}', end="")
        if episode % 100 == 0:
            print(f'\repisode {episode}, mean: {np.mean(history)}, variance: {np.sqrt(np.var(history))}, min: {np.min(history)}, max: {np.max(history)}, eps: {eps}')
        if episode % 10 == 0 and np.mean(history) >= 180:
            agent.save(path=f'agent_{episode}, score_{np.mean(history)}, variance_{np.sqrt(np.var(history))}, min_{np.min(history)}, max_{np.max(history)}.pkl')