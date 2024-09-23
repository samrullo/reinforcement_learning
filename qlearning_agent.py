import numpy as np
from common.utils import greedy_probs
from common.grid_world import GridWorld, State, Action
from collections import defaultdict, deque
from tqdm import tqdm
import matplotlib

matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg', depending on your setup


class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1

        self.Q = defaultdict(lambda: 0)

        random_actions = {a: 1 / len(Action) for a in Action}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)

    def get_action(self, state: State):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def update(self, state, action, reward, next_state, done):
        if done:
            next_q = 0
        else:
            next_qs = [self.Q[next_state, a] for a in Action]
            next_q = max(next_qs)

        target = reward + self.gamma * next_q
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

        self.b[state] = greedy_probs(self.Q, state, self.epsilon)
        self.pi[state] = greedy_probs(self.Q, state, 0)


if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningAgent()

    episodes = 10000

    for e in tqdm(range(episodes)):
        state = env.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            if done:
                break
            state = next_state
    env.render_q(agent.Q)
