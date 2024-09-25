import numpy as np
import gym

import matplotlib

matplotlib.use('TkAgg')

env = gym.make('CartPole-v0', render_mode="human")
state = env.reset()
done = False

while not done:
    env.render()
    action = np.random.choice([0, 1])
    next_state, reward, done, _, info = env.step(action)
    print(f"next state is : {next_state}")

env.close()
