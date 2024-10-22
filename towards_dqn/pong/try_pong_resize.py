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

import torch
from torchvision.transforms import transforms

img_pt = torch.from_numpy(img_arr)
img_pt = transforms.Resize((110,84))(img_pt.unsqueeze(0))
img_pt = img_pt.squeeze(0)
img_pt = img_pt[-84:,:]

new_img=img_pt.numpy()




fig, ax = plt.subplots(1, 2)
ax[0].imshow(Image.fromarray(img_arr))
ax[0].set_title("Original")
ax[1].imshow(Image.fromarray(new_img))
ax[1].set_title("Truncated")
plt.show()
