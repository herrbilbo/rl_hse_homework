import gym
import random
from collections import namedtuple, deque
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
import pybullet_envs


class actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.fc4 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = F.tanh(self.fc3(x))
        sigma = F.softplus(self.fc4(x))
        return mu, sigma

        
class critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
        
def weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)
    

class replay_buffer:
    def __init__(self, max_size, batch_size, num_batches, device):
        self.max_size = max_size
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.memory = [None] * self.max_size
        self.device = device
        self.tail = 0
        self.len = 0

    def push(self, cur_transition):
        self.memory[self.tail] = cur_transition
        self.len = min(self.len + 1, self.max_size)
        self.tail = (self.tail + 1) % self.max_size

    def sample(self):
        batches = []
        for i in range(self.num_batches):
            batch = transition(*zip(*random.sample(self.memory, self.batch_size)))
            
            batches.append(transition(state=torch.FloatTensor(batch.state).to(agent.device),
                                        action=torch.FloatTensor(batch.action).to(agent.device), 
                                        next_state=torch.FloatTensor(batch.next_state).to(agent.device), 
                                        reward=torch.FloatTensor(batch.reward).to(agent.device), 
                                        done=torch.tensor(batch.done).to(agent.device)))
        
        self.memory = [None] * self.max_size
        self.tail = 0
        self.len = 0
        return batches
        
    def __len__(self):
        return self.len
        
        
class ppo:
    def __init__(self, env, batch_size, buffer_size, num_batches, actor_lr, critic_lr, gamma, eps):
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.low = env.action_space.low[0]
        self.high = env.action_space.high[0]
        
        self.gamma = gamma
        self.eps = eps
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = actor(self.state_dim, self.action_dim).to(self.device).apply(weights)
        self.old_actor = actor(self.state_dim, self.action_dim).to(self.device).apply(weights)
        self.critic = critic(self.state_dim, self.action_dim).to(self.device).apply(weights)
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.buffer_size = buffer_size
        self.replay_buffer = replay_buffer(self.buffer_size, self.batch_size, self.num_batches, self.device)
        
    def push(self, transition):
        self.replay_buffer.push(transition)

    def update(self):        
        if len(self.replay_buffer) == self.buffer_size:
            self.old_actor.load_state_dict(self.actor.state_dict())
            
            batches = self.replay_buffer.sample()
            
            for batch in batches:
                V = self.critic(batch.state)
                
                with torch.no_grad():
                    V_next = self.critic(batch.next_state)
                    
                Q = batch.reward.unsqueeze(1) + self.gamma * V_next
                
                with torch.no_grad():
                    Advantage = (Q - V)
                
                mu, sigma = self.actor(batch.state)
                log_proba = Normal(mu, sigma).log_prob(batch.action).sum(dim=1, keepdims=True)
                
                with torch.no_grad():
                    old_mu, old_sigma = self.old_actor(batch.state)
                old_log_proba = Normal(old_mu, old_sigma).log_prob(batch.action).sum(dim=1, keepdims=True)
                
                r = torch.exp(log_proba - old_log_proba)
                
                actor_loss = -torch.min(r * Advantage, torch.clamp(r, 1.0 - self.eps, 1.0 + self.eps) * Advantage).mean()
                critic_loss = ((Q - V) ** 2).mean()
        
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                
                actor_loss.backward()
                critic_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)

                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
    def act(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mu, sigma = self.actor(state)
            
        norm = Normal(mu, sigma)
        return torch.clamp(norm.sample().view(-1), min=self.low, max=self.high).cpu().numpy()
        
    def save(self, name=''):
        torch.save(self.actor.state_dict(), name + "_actor.pickle")

            
if __name__ == '__main__':
    transition = namedtuple('transition', ('state', 'action', 'next_state', 'reward', 'done'))
    env = gym.make("HalfCheetahBulletEnv-v0")
    
    print(env.observation_space.shape)
    print(env.action_space.shape)
    print(env.action_space.low)
    print(env.action_space.high)
    
    seed = 228
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    max_episodes = 5000

    agent = ppo(env, batch_size=128, buffer_size=1000, num_batches=64, actor_lr=1e-4, critic_lr=1e-3, gamma=0.93, eps=0.1)
    
    print(agent.device)

    history = deque(maxlen=100)
    for episode in range(max_episodes):
        state = env.reset()
        total = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total += reward
            agent.push(transition(state, action, next_state, reward, done))
            
            state = next_state
            agent.update()
            
        history.append(total)
        print(f'\repisode {episode}, total: {total}, mean: {np.mean(history)}, variance: {np.sqrt(np.var(history))}, min: {np.min(history)}, max: {np.max(history)}', end="")
        if episode % 50 == 0:
            print(f'\repisode {episode}, total: {total}, mean: {np.mean(history)}, variance: {np.sqrt(np.var(history))}, min: {np.min(history)}, max: {np.max(history)}')

        if episode % 50 == 0:
            agent.save(name=f'episode_{episode}')