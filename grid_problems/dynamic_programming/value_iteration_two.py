import numpy as np
from common.grid_world import GridWorld, State
from collections import defaultdict


def value_iter_one_step(V: defaultdict, env: GridWorld, gamma: float = 0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
        action_values = []
        for action in env.actions:
            next_state = env.next_state(state, action)
            recieved_reward = env.reward(state, action, next_state)
            action_values.append(recieved_reward + gamma * V[next_state])
        V[state] = max(action_values)
    return V


def value_iter(env: GridWorld, gamma: float = 0.9, thresh: float = 0.001):
    V = defaultdict(lambda: 0)
    while True:
        old_V = V.copy()

        V = value_iter_one_step(V, env, gamma)
        delta = max([abs(V[state] - old_V[state]) for state in env.states()])

        if delta < thresh:
            break
    return V


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    thresh = 0.001

    V = value_iter(env, gamma, thresh)
    from grid_problems.dynamic_programming.iterative_policy_improvement import greedy_policy

    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi, print_value=True)
