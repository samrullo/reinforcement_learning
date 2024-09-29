import gym
import random
import copy
import numpy as np
from dezero import Model
import dezero.functions as F
import dezero.layers as L
from dezero import optimizers
from collections import deque
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg', depending on your setup


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=self.buffer_size)

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def get_batch(self):
        # list_of_data is a list of tuples, where each tuple is (state,action,reward,next_state,done)
        list_of_data = random.sample(self.buffer, self.batch_size)

        # stack states vertically so that the resulting statees will have (batch_size,state_dim) shape
        states = np.stack([data[0] for data in list_of_data])

        # stack actions, this will have a shape of (batch_size,)
        actions = np.stack([data[1] for data in list_of_data])

        # stack rewards, this will also have a shape of (batch_size,)
        rewards = np.stack([data[2] for data in list_of_data])

        # stack next states vertically. this will have a shape of (batch_size, state_dim)
        next_states = np.stack([data[3] for data in list_of_data])

        # stack done or not done. this will be a list of booleans with shape (batch_size,)
        dones = np.stack([data[4] for data in list_of_data])

        return states, actions, rewards, next_states, dones


class Qnet(Model):
    def __init__(self, action_size: int):
        super().__init__()
        self.l1 = L.Linear(128)
        self.l2 = L.Linear(128)
        self.l3 = L.Linear(action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        out = self.l3(x)
        return out


class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)

        self.qnet = Qnet(self.action_size)
        self.qnet_target = Qnet(self.action_size)

        self.optimizer = optimizers.Adam(self.lr)
        self.optimizer.setup(self.qnet)

    def sync_qnet(self):
        self.qnet_target = copy.deepcopy(self.qnet)

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            if isinstance(state, tuple):
                state = np.array(state)
            # first transform state from (4,) to (1,4) so that qnet can process it
            state = state[np.newaxis, :]

            # qnet will return q_value for each action, so this will return (1,2) array
            q_values = self.qnet(state)

            # return the action with maximum q value
            return q_values.data.argmax()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer) < self.batch_size:
            return

        # sample a batch from the replay buffer which stores agent's experience
        states, actions, rewards, next_states, dones = self.replay_buffer.get_batch()

        # Q learning is about updating Q function to bring it closer to the TD target which
        # is calculated as R + gamma * (max of Q_next)
        next_q_values = self.qnet_target(next_states)
        next_q_max = next_q_values.max(axis=1)
        next_q_max.unchain()
        targets = rewards + (1 - dones) * self.gamma * next_q_max

        q_values = self.qnet(states)
        # we access Q values corresponding to actions
        action_qs = q_values[np.arange(self.batch_size), actions]

        loss = F.mean_squared_error(targets, action_qs)

        self.qnet.cleargrads()
        loss.backward()
        self.optimizer.update()


if __name__ == "__main__":
    episodes = 300
    sync_interval = 20
    env = gym.make('CartPole-v0')
    agent = DQNAgent()
    reward_history = []

    for episode in tqdm(range(episodes)):
        state, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            state = next_state

            total_reward += reward

        if episode % sync_interval == 0:
            agent.sync_qnet()
        reward_history.append(total_reward)
        # print(f"Gained total reward of {total_reward} at episode {episode + 1}")

    # === Plot ===
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()

    # === Play CartPole ===
    env = gym.make('CartPole-v0', render_mode="human")
    agent.epsilon = 0  # greedy policy
    state, info = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.get_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        state = next_state
        total_reward += reward
        env.render()
    print('Total Reward:', total_reward)
