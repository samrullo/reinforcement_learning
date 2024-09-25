import numpy as np
import gym

import matplotlib

matplotlib.use('TkAgg')

env = gym.make('CartPole-v0', render_mode="human")
state = env.reset()
done = False
total_reward=0

while not done:
    env.render()
    action = np.random.choice([0, 1])
    next_state, reward, done, _, info = env.step(action)
    total_reward+=reward
    print(f"next state is : {next_state}")

print(f"total reward : {total_reward}")
env.close()
