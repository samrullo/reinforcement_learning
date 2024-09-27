from collections import deque
import gym

env = gym.make('Pong-v4', render_mode="human",obs_type="grayscale")

# Reset the environment to start a new game
state = env.reset()

print(f"state is a tuple with size {len(state)}")
print(f"first element of the state is an array with shape : {state[0].shape}")
print(f"action space is {env.action_space}")
print(f"observation space : {env.observation_space}")

done = False
cnt = 0
while not done:
    cnt += 1
    print(f"Iteration : {cnt}")
    # Render the game (optional, comment this out if not needed)
    env.render()

    # Choose a random action (you will later replace this with your DQN model's action)
    action = env.action_space.sample()
    print(f"Picked up action : {action}")

    # Perform the action
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    print(f"next_state shape : {next_state.shape}, reward : {reward}, done : {done}, info : {info}")

    # Update the current state
    state = next_state

    # if cnt > 100:
    #     break

# Close the environment after the game is done
env.close()
