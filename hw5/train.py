import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import pybullet_envs
import random
import gym
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


class critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(critic, self).__init__()
        self.linear1 = nn.Linear(state_size + action_size, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state, action):
        res = torch.cat((state, action), dim=1)
        res = F.relu(self.linear1(res))
        res = F.relu(self.linear2(res))
        res = self.linear3(res)
        return res
        
        
def init_weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)

        
class replay_buffer:
    def __init__(self, max_size, batch_size):
        self.max_size = max_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_size)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self):
        return list(zip(*random.sample(self.buffer, self.batch_size)))

    def __len__(self):
        return len(self.buffer)


class ddpg:
    def __init__(self, state_dim, action_dim, buffer_size, batch_size, gamma, tau, actor_lr, critic_lr):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.gamma = gamma
        self.tau = tau
        
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        
        self.actor = actor(self.state_dim, self.action_dim).to(self.device).apply(init_weights)
        self.critic = critic(self.state_dim, self.action_dim).to(self.device).apply(init_weights)
        
        self.actor_target = actor(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = critic(self.state_dim, self.action_dim).to(self.device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.hard_update()

        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer(self.buffer_size, self.batch_size)

    def act(self, state):
        state = torch.tensor(state).to(self.device).float()
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy()
        return action
        
    def update(self, transition):
        self.replay_buffer.push(transition)
        
        if len(replay_buffer) >= self.batch_size:
            batch = self.replay_buffer.sample()
            
            states, actions, rewards, next_states, dones = batch

            states = torch.tensor(states).to(self.device).float()
            next_states = torch.tensor(next_states).to(self.device).float()
            rewards = torch.tensor(rewards).to(self.device).float()
            actions = torch.tensor(actions).to(self.device).float()
            dones = torch.tensor(dones).to(self.device).int()

            next_actions = self.actor_target(next_states)
            Q = self.critic_target(next_states, next_actions).detach()
            Q = rewards.unsqueeze(1) + self.gamma * ((1 - dones).unsqueeze(1)) * Q

            critic_loss = ((self.critic(states, actions) - Q)**2).mean()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actions_pred = self.actor(states)
            
            actor_loss = -self.critic(states, actions_pred).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update()
            
    def hard_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path=''):
        torch.save(self.actor.state_dict(), path + '_actor.pkl')
        torch.save(self.critic.state_dict(), path + '_critic.pkl')


if __name__== "__main__":
    env = gym.make("AntBulletEnv-v0")
    
    print(env.observation_space.shape)
    print(env.action_space.shape)
    print(env.action_space.low)
    print(env.action_space.high)

    state_dim = 28
    action_dim = 8

    seed = 239
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    buffer_size = 1000000
    batch_size = 64
    gamma = 0.95
    tau = 0.001
    episodes = 5000
    actor_lr = 1e-4
    critic_lr = 1e-3
    std = 1
    std_min = 0.1
    
    agent = ddpg(state_dim=state_dim, action_dim=action_dim, buffer_size=buffer_size, batch_size=batch_size, gamma=gamma, tau=tau, actor_lr=actor_lr, critic_lr=critic_lr)
    
    print(agent.device)
                      
    history = deque(maxlen=100)
    
    for episode in range(episodes):
        state = env.reset()
        total = 0
        done = False
        while not done:
            action = agent.act(state)
            
            noise = np.random.normal(loc=0.0, scale=std, size=action.shape)
            action = action + noise
            action = np.clip(action, -1.0, 1.0)
            
            next_state, reward, done, _ = env.step(action)
            transition = state, action, reward, next_state, done
            agent.update(transition)
            total += reward
            state = next_state

        history.append(total)
        std = max(std - 0.005, std_min)
        print(f'\repisode {episode}, total: {total}, mean: {np.mean(history)}, variance: {np.sqrt(np.var(history))}, min: {np.min(history)}, max: {np.max(history)}, std: {std}', end="")
        if episode % 10 == 0:
            print(f'\repisode {episode}, total: {total}, mean: {np.mean(history)}, variance: {np.sqrt(np.var(history))}, min: {np.min(history)}, max: {np.max(history)}, std: {std}')
        if episode % 10 == 0:
            agent.save(f'episode_{episode}')