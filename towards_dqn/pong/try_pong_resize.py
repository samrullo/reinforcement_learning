import gym

env = gym.make('Pong-v4', render_mode="rgb_array",obs_type="grayscale")

# Reset the environment to start a new game
state = env.reset()

print(f"state is a tuple with size {len(state)}")
print(f"first element of the state is an array with shape : {state[0].shape}")
print(f"action space is {env.action_space}")
print(f"observation space : {env.observation_space}")

from PIL import Image
import matplotlib.pyplot as plt

img_arr = state[0]
fig, ax = plt.subplots(1, 2)
ax[0].imshow(Image.fromarray(img_arr))
ax[0].set_title("Original")
ax[1].imshow(Image.fromarray(img_arr[35:-15, :]))
ax[1].set_title("Truncated")
plt.show()
