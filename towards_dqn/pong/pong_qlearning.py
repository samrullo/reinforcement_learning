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


class PongQnet(nn.Module):
    def __init__(self, action_size, *args, **kwargs):
        self.action_size = action_size
        super().__init__(*args, **kwargs)

        # the input will have a shape (C,H,W) where C=4
        initial_width = 64
        kernel_width = 3
        stride_width = 2
        padding = 1
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=(kernel_width, kernel_width),
                               stride=(stride_width, stride_width), padding=(padding, padding))
        new_width = (initial_width + 2 * padding - kernel_width) // stride_width + 1
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(kernel_width, kernel_width),
                               stride=(stride_width, stride_width), padding=(padding, padding))
        new_width = (new_width + 2 * padding - kernel_width) // stride_width + 1
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(kernel_width, kernel_width),
                               stride=(stride_width, stride_width), padding=(padding, padding))
        new_width = (new_width + 2 * padding - kernel_width) // stride_width + 1
        self.l1 = nn.Linear(new_width * new_width * 256, 512)
        self.out = nn.Linear(512, self.action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
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
        states = np.stack([item[0] for item in data])
        actions = np.stack([item[1] for item in data])
        rewards = np.stack([item[2] for item in data])
        next_states = np.stack([item[3] for item in data])
        dones = np.stack([item[4] for item in data])
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


def preprocess_state(state: np.ndarray):
    # state is array with length (210,160)
    # return torch tensor with shape (1,H,W)
    return torch.from_numpy(state[35:-15, :]).unsqueeze(0)


def concat_four_frames(list_of_four_frames: deque):
    return torch.cat(list(list_of_four_frames), dim=0)


resize = Resize(64)
IMAGE_SIZE = 64


def preprocess_input_tensors(input_tensors):
    resized_tensors = resize(input_tensors)
    return resized_tensors / 255.0 - 1.0


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
        self.lr = 0.0005
        self.epsilon = 1
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

    def update_epsilon(self):
        self.epsilon = 1 - self.epsilon / self.last_episode_for_epsilon_warmup

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            q_values = self.qnet(state)

            # assuming q_values has shape (1,action_size)
            q_vals = q_values.squeeze(dim=0).numpy()
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
            rewards).float().to(device), torch.from_numpy(next_states).float().to(device), torch.from_numpy(np.array(dones)).float().to(device)

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


if __name__ == "__main__":
    episodes = 300
    sync_interval = 20
    env = gym.make('Pong-v4', render_mode="rgb_array", obs_type="grayscale")
    agent = PongDQN()
    four_frame_buff = FourFrameBuffer()
    weights_folder=pathlib.Path.cwd()/"weights"
    reward_history = []

    for episode in tqdm(range(episodes)):
        state, info = env.reset()
        state = preprocess_state(state).to(device)
        state = preprocess_input_tensors(state).to(device)
        four_frame_buff.add(state)
        state = concat_four_frames(four_frame_buff.buffer).to(device)
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = preprocess_state(next_state).to(device)
            next_state = preprocess_input_tensors(next_state).to(device)
            four_frame_buff.add(next_state)

            done = terminated or truncated

            next_state = concat_four_frames(four_frame_buff.buffer).to(device)
            agent.update(state, action, reward, next_state, done)

            state = next_state

            total_reward += reward


        if episode % sync_interval == 0:
            agent.sync_qnet()

            # save model weights
            torch.save(agent.qnet.state_dict(),str(weights_folder/f"qnet_weights_{episode}.pth"))
            torch.save(agent.qnet_target.state_dict(),str(weights_folder/f"qnet_target_weights_{episode}.pth"))

        reward_history.append(total_reward)