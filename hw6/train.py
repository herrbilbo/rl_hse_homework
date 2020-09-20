from gym import make
from agent import Agent
import torch


if __name__ == "__main__":
    env = make("MountainCar-v0")
    agent = Agent(state_dim=2, action_dim=3)
    episodes = 100
    visit_count = 0
    

    for i in range(episodes):
        state = env.reset()
        steps = 0
        done = False
        max_state = -200
        while not done:
            #env.render()
            max_state = max(state[0], max_state)
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            steps += 1
            agent.update((state, action, next_state, reward, done))
            state = next_state
        if steps < 200:
            visit_count += 1
            print("Visited target state at episode", i, 'max state =', max_state)
            #print("Visited target state at episode", i)
        else:
            print("Episode", i, 'max state =', max_state)
            #print("Episode", i)
    print()
    print("Total visit count:", visit_count)
