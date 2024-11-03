import numpy as np
from common.grid_world import State, Action, GridWorld
from common.utils import greedy_probs
from collections import defaultdict
from collections import deque
from tqdm import tqdm
import matplotlib

matplotlib.use('TkAgg')  # or 'Qt5Agg', 'Agg', depending on your setup


class SarsaPolicyOffAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.8
        self.epsilon = 0.1

        random_actions = {a: 1 / len(Action) for a in Action}
        self.pi = defaultdict(lambda: random_actions)
        self.b = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)

        self.memory = deque(maxlen=2)

    def get_action(self, state: State):
        action_probs = self.b[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def reset(self):
        self.memory.clear()

    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]

        if done:
            next_q = 0
            rho = 1
        else:
            next_q = self.Q[next_state, next_action]
            rho = self.pi[next_state][next_action] / self.b[next_state][next_action]

        target = rho * (reward + self.gamma * next_q)

        self.Q[state, action] += self.alpha * (target - self.Q[state, action])

        self.b[state] = greedy_probs(self.Q, state, self.epsilon)
        self.pi[state] = greedy_probs(self.Q, state, 0)


if __name__ == "__main__":
    env = GridWorld()
    agent = SarsaPolicyOffAgent()
    episodes = 10000

    for e in tqdm(range(episodes)):
        #print(f"Start episode {e + 1}")
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            # this should come before done evaluation in next line
            agent.update(state, action, reward, done)

            if done:
                # forgot this in the first attempt
                agent.update(next_state, None, None, None)
                break
            state = next_state

    env.render_q(agent.Q)
