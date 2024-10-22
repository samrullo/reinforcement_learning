import gym
from tqdm import tqdm
import torch
from torchvision.transforms import ToTensor, Resize, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

import numpy as np
from collections import deque, defaultdict
import random
import pathlib

# Set the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"will use device : {device}")


class PongQnet(nn.Module):
    def __init__(self, action_size, *args, **kwargs):
        self.action_size = action_size
        super().__init__(*args, **kwargs)

        # the input will have a shape (C,H,W) where C=4
        initial_width = 84
        out_channels = [16, 32]
        kernels = [8, 4]
        strides = [4, 2]
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=out_channels[0], kernel_size=kernels[0],
                               stride=strides[0], padding=0)
        new_width = (initial_width - kernels[0]) // strides[0] + 1
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=out_channels[1], kernel_size=4,
                               stride=2, padding=0)
        new_width = (new_width + - kernels[1]) // strides[1] + 1
        self.l1 = nn.Linear(new_width * new_width * out_channels[-1], 256)
        self.out = nn.Linear(256, self.action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        B = x.size(0)
        x = x.view(B, -1)
        x = F.relu(self.l1(x))
        out = self.out(x)
        return out


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.buffer = deque(maxlen=self.buffer_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def get_batch(self):
        data = random.sample(self.buffer, self.batch_size)
        states = np.stack([item[0].cpu().numpy() for item in data])
        actions = np.stack([item[1] for item in data])
        rewards = np.stack([item[2] for item in data])
        next_states = np.stack([item[3].cpu().numpy() for item in data])
        dones = np.stack([item[4] for item in data])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def preprocess_state(state: np.ndarray):
    # state is array with length (210,160)
    # first resize it to (110,84) and then crop (84,84) region
    # return torch tensor with shape (1,H,W)
    img_pt = torch.from_numpy(state)
    img_pt = transforms.Resize((110, 84))(img_pt.unsqueeze(0))
    img_pt = img_pt[:,-84:, :]
    return img_pt


def concat_four_frames(list_of_four_frames: deque):
    thedevice = list_of_four_frames[0].device
    return torch.cat([frame.to(thedevice) for frame in list(list_of_four_frames)], dim=0)


IMAGE_SIZE = 84


def preprocess_input_tensors(input_tensors):
    return input_tensors / 255.0 - 1.0


class FourFrameBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=4)
        # populate first 3 frames with zero tensors
        for i in range(3):
            self.buffer.append(torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE)))

    def add(self, new_frame: torch.tensor):
        self.buffer.append(new_frame)


class PongDQN:
    def __init__(self):
        self.gamma = 0.98
        self.alpha = 0.8
        self.lr = 0.01
        self.epsilon = 1
        self.initial_epsilon = 1
        self.last_epsilon = 0.1
        self.action_size = 6
        self.last_episode_for_epsilon_warmup = 50

        self.buffer_size = 10000
        self.batch_size = 32
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)

        self.four_frame_buffer = FourFrameBuffer()

        self.qnet = PongQnet(self.action_size).to(device)
        self.qnet_target = PongQnet(self.action_size).to(device)

        self.optimizer = Adam(self.qnet.parameters(), lr=self.lr)
        self.mse_loss = nn.MSELoss()

    def update_epsilon(self, episode: int):
        if episode <= self.last_episode_for_epsilon_warmup:
            self.epsilon = 1 + (
                    (self.last_epsilon - self.initial_epsilon) / self.last_episode_for_epsilon_warmup) * episode

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = self.qnet(state)

            # assuming q_values has shape (1,action_size)
            q_vals = q_values.squeeze(dim=0).cpu().detach().numpy()
            return np.argmax(q_vals)

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        # sample a batch from the replay buffer which stores agent's experience
        states, actions, rewards, next_states, dones = self.replay_buffer.get_batch()

        states, actions, rewards, next_states, dones = torch.from_numpy(states).float().to(device), torch.from_numpy(
            actions).long().to(device), torch.from_numpy(
            rewards).float().to(device), torch.from_numpy(next_states).float().to(device), torch.from_numpy(
            np.array(dones)).float().to(device)

        # Q learning is about updating Q function to bring it closer to the TD target which
        # is calculated as R + gamma * (max of Q_next)
        next_q_values = self.qnet_target(next_states)
        next_q_max, _idx = next_q_values.max(dim=1)
        next_q_max.detach()
        targets = rewards + (1 - dones) * self.gamma * next_q_max

        q_values = self.qnet(states)
        # we access Q values corresponding to actions
        action_qs = q_values[np.arange(self.batch_size), actions]

        loss = self.mse_loss(targets, action_qs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


if __name__ == "__main__":
    episodes = 1000
    sync_interval = 20
    env = gym.make('Pong-v4', render_mode="rgb_array", obs_type="grayscale")
    agent = PongDQN()
    four_frame_buff = FourFrameBuffer()
    weights_folder = pathlib.Path.cwd() / "weights"
    weights_folder.mkdir(parents=True, exist_ok=True)
    reward_history = []
    loss_history = []

    progress_bar = tqdm(range(episodes), desc="Training progress")
    for episode in progress_bar:
        state, info = env.reset()
        state = preprocess_state(state).to(device)
        state = preprocess_input_tensors(state).to(device)
        four_frame_buff.add(state)
        state = concat_four_frames(four_frame_buff.buffer).to(device)
        done = False
        total_reward = 0
        episode_loss = 0
        loss_count = 0

        while not done:
            action = agent.get_action(state.unsqueeze(0))
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_state(next_state).to(device)
            next_state = preprocess_input_tensors(next_state).to(device)
            four_frame_buff.add(next_state)

            done = terminated or truncated

            next_state = concat_four_frames(four_frame_buff.buffer).to(device)
            loss = agent.update(state, action, reward, next_state, done)

            if loss is not None:
                episode_loss += loss
                loss_count += 1

            state = next_state

            total_reward += reward

        if episode % sync_interval == 0:
            agent.sync_qnet()

            # save model weights
            torch.save(agent.qnet.state_dict(), str(weights_folder / f"qnet_weights_{episode}.pth"))
            torch.save(agent.qnet_target.state_dict(), str(weights_folder / f"qnet_target_weights_{episode}.pth"))

        reward_history.append(total_reward)

        # Average loss for the episode
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        loss_history.append(avg_loss)

        agent.update_epsilon(episode)
        progress_bar.set_postfix({
            "avg_loss": f"{avg_loss:.4f}",
            "total_reward": f"{total_reward:.2f}",
            "epsilon": f"{agent.epsilon:.2f}"
        })
