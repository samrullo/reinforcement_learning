import gym

env = gym.make('CartPole-v0')
observation = env.reset()
action = 0  # Example action (e.g., moving left)

initial_state,info=env.reset()
print(f"Initial state : {initial_state}")

# Perform the action in the environment
next_observation, reward, terminated, truncated, info = env.step(action)


print(f"Next Observation: {next_observation}")
print(f"Reward: {reward}")
print(f"Done , terminated: {terminated}")
print(f"Truncated : {truncated}")
print(f"Info: {info}")
