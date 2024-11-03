import numpy as np
from collections import defaultdict
from common.grid_world import State, GridWorld, Action
import matplotlib

matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg', depending on your setup


class MCAgent:
    def __init__(self):
        self.alpha = 0.8
        self.gamma = 0.9
        self.epsilon = 0.1

        self.action_size = 4
        random_actions = {a: 1 / self.action_size for a in range(self.action_size)}

        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)

        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add_to_memory(self, state: State, action: int, reward: float):
        self.memory.append((state, action, reward))

    def reset(self):
        self.memory.clear()

    def eval(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = reward + self.gamma * G
            key = (state, action)
            self.Q[key] += self.alpha * (G - self.Q[key])
            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)


def greedy_probs(Q, state: State, epsilon: float):
    qs = [Q[state, a] for a in Action]
    max_action = np.argmax(qs)
    action_probs = {a: epsilon / len(Action) for a in Action}
    action_probs[max_action] += 1 - epsilon
    return action_probs


if __name__ == "__main__":
    env = GridWorld()
    agent = MCAgent()

    for e in range(10000):
        print(f"Start episode {e + 1}")
        state = env.reset()
        agent.reset()
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.add_to_memory(state, action, reward)
            if done:
                agent.eval()
                break
            state = next_state
    env.render_q(agent.Q)
