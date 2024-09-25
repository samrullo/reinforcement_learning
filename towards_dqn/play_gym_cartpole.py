import gym

env = gym.make('CartPole-v0')

state = env.reset()
print("initial state is : ", state)

action_space = env.action_space
print("action space is : ", action_space)

action = 0
next_state, reward, done, something, info = env.step(action)
print("next state after an action 0 : ", next_state)
print(f"reward is : {reward}, done is : {done}")
