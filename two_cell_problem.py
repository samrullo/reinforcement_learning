import numpy as np
from enum import IntEnum


class Action(IntEnum):
    LEFT = 0
    RIGHT = 1


class GridEnv:
    def __init__(self):
        # initialize all states
        self.height = 1
        self.width = 2

        states = []
        for h in range(self.height):
            for w in range(self.width):
                states.append((h, w))
        self.states = states

        self.actions = [a for a in Action]

        self.reward_map = np.zeros((self.height, self.width))
        self.reward_map[0, 1] = 1

    def state_transition(self, state, action: Action):
        if action == Action.LEFT:
            next_state = (state[0], state[1] - 1)
        else:
            next_state = (state[0], state[1] + 1)

        x, y = next_state
        if x < 0 or y < 0 or x > self.width or y > self.height:
            next_state = state
        return next_state

    def get_reward(self, state, action, next_state):
        # next_state equals state means the agent bumped against walls and need to be punished
        if next_state == state:
            return -1.0
        else:
            return self.reward_map[next_state]

    def get_states(self):
        return self.states


env = GridEnv()
gamma = 0.9

V = {state: 0.0 for state in env.get_states()}
action_probs = {a: 1 / len(env.actions) for a in env.actions}
# move_to_right = {Action.RIGHT: 1.0, Action.LEFT: 0.0}
# move_to_left = {Action.RIGHT: 0.0, Action.LEFT: 1.0}
# pi = {(0, 0): move_to_right, (0, 1): move_to_left}
pi = {state: action_probs for state in V}

max_update_diff = 1e+6
temp = 1e-6
thresh = 1e-4
cnt = 0
# continue until value updates don't converge to a small threshold
while max_update_diff > thresh and cnt < 1000000:
    # calculate new Values using Bellman equation
    V_old = V.copy()

    for state in env.get_states():
        state_value = 0
        for action in env.actions:
            next_state = env.state_transition(state, action)
            action_prob = pi[state][action]
            reward = env.get_reward(state, action, next_state)
            state_value += action_prob * (reward + gamma * V[next_state])
        V[state] = state_value
    max_update_diff = 0
    for state in V:
        max_update_diff = max(max_update_diff, abs(V[state] - V_old[state]))
    cnt += 1

print(V)
