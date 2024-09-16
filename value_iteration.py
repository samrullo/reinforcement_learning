import numpy as np
from common.grid_world import GridWorld, State, Action
from collections import defaultdict


def value_iter_one_step(V: defaultdict, env: GridWorld, gamma: float = 0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
        action_values = []
        for action in env.actions:
            next_state = env.next_state(state, action)
            thereward = env.reward(state, action, next_state)
            action_values.append(thereward + gamma * V[next_state])
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
    reward_map = np.zeros((7, 8))
    wall_states = [State(1, 5), State(5, 1), State(5, 5), State(3, 6)]
    bomb_states = [State(3, 2), State(4, 2), State(4, 3), State(4, 4), State(3, 4), State(1, 7)]
    goal_state = State(6, 7)
    apple_states = [goal_state]
    env = GridWorld(reward_map, wall_states, apple_states, bomb_states, goal_state,6.0)
    gamma = 0.9
    thresh = 0.001

    V = value_iter(env, gamma, thresh)
    from iterative_policy_improvement import greedy_policy

    pi = greedy_policy(V, env, gamma)
    env.render_v(V, pi, print_value=True)
