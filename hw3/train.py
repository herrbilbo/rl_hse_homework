import gym
import random
from collections import namedtuple, deque
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
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

        
class critic(nn.Module):
    def __init__(self, state_dim):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
        
def weights(layer):
    if isinstance(layer, nn.Linear):
        nn.init.xavier_normal_(layer.weight)


class aql:
    def __init__(self, state_dim, low, high, gamma, actor_lr, critic_lr, batch_size):
        self.state_dim = state_dim
        self.low = low
        self.high = high
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.actor = actor(self.state_dim).to(self.device).apply(weights)
        self.critic = critic(self.state_dim).to(self.device).apply(weights)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)


    def update(self, batch):
        critic_pred = self.critic.forward(batch.state)
        critic_next = torch.zeros(critic_pred.size(), device=self.device)
        
        with torch.no_grad():
            critic_next[~batch.done] = self.critic.forward(batch.next_state)[~batch.done]
            
        target = batch.reward.unsqueeze(1) + self.gamma * critic_next
        
        advantage = target - critic_pred
        
        actor_loss = -(advantage * batch.log_proba).mean()
        critic_loss = ((advantage) ** 2).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(self.device)
        mu, sigma = self.actor(state)
        norm = Normal(mu, sigma)
        action = torch.clamp(norm.sample().view(1), min=self.low, max=self.high)
        return np.array([action.item()]), norm.log_prob(action)
        
    def save(self, name=''):
        torch.save(self.actor.state_dict(), name + " actor.pickle")

if __name__ == '__main__':

    transition = namedtuple('transition', ('state', 'action', 'next_state', 'reward', 'done', 'log_proba'))
    
    env = gym.make('Pendulum-v0')
    
    seed = 228
    env.seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    max_episodes = 20000

    agent = aql(state_dim=3, low=-2, high=2, gamma=0.95, actor_lr=1e-4, critic_lr=1e-3, batch_size=40)

    history = deque(maxlen=100)
    trajectory = []
    for episode in range(max_episodes):
        state = env.reset()
        total = 0
        done = False
        while not done:
            action, log_proba = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            total += reward
            
            # let's scale reward to [-1; 1]
            reward = (reward + 8.0) / 8.0
            trajectory.append( (state, action, next_state, reward, done, log_proba) )
            
            if len(trajectory) == agent.batch_size:
                states, actions, rewards, next_actions, dones, log_probas = zip(*trajectory)
                batch = transition(state=torch.FloatTensor(states).to(agent.device),
                           action=torch.FloatTensor(actions).to(agent.device),
                           next_state=torch.FloatTensor(rewards).to(agent.device),
                           reward=torch.FloatTensor(next_actions).to(agent.device),
                           done=torch.tensor(dones).to(agent.device),
                           log_proba=torch.cat(log_probas).to(agent.device))
                           
                agent.update(batch)
                trajectory = []
            state = next_state
            
        history.append(total)
        print(f'\repisode {episode}, mean: {np.mean(history)}, variance: {np.sqrt(np.var(history))}, min: {np.min(history)}, max: {np.max(history)}', end="")
        if episode % 100 == 0:
            print(f'\repisode {episode}, mean: {np.mean(history)}, variance: {np.sqrt(np.var(history))}, min: {np.min(history)}, max: {np.max(history)}')

        if episode % 100 == 0:
            agent.save(name=f'episode_{episode}')