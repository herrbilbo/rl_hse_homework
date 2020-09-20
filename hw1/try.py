import gym
from agent import Agent

agent = Agent()
env = gym.make('MountainCar-v0')
env.seed(228)

def average_reward(agent, env, n=20):
    res = 0
    for _ in range(n):
        state = env.reset()
        reward = 0
        done = False
        while not done:
            action = agent.act(state)
            state, r, done, _ = env.step(action)
            reward += r
        res += reward
    return res / n

print(average_reward(agent, env))

state = env.reset()
reward = 0
done = False
while not done:
    env.render()
    action = agent.act(state)
    state, r, done, _ = env.step(action)
    reward += r
env.close()
