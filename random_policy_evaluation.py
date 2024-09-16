import numpy as np
from collections import defaultdict
from common.grid_world import GridWorld, Action, State

random_actions = {action: 1 / len(Action) for action in Action}
random_policy = defaultdict(lambda: random_actions)
v = defaultdict(lambda: 0)

grid = GridWorld()
gamma = 0.9
thresh = 0.001

cnt = 0
while True:
    v_old = v.copy()

    # one step evaluation
    for state in grid.states():
        if state == grid.goal_state:
            v[state] = 0
            continue
        state_value = 0
        for action in grid.actions:
            next_state = grid.next_state(state, action)
            reward = grid.reward(state, action, next_state)
            action_prob = random_policy[state][action]
            state_value += action_prob * (reward + gamma * v[next_state])
        v[state] = state_value

    # find the maximum difference between newly evaluated state values and previously evaluated state values
    delta = 0
    for state in v.keys():
        t = abs(v[state] - v_old[state])
        if delta < t:
            delta = t

    cnt += 1
    print(f"Iteration {cnt}, delta is {delta}")
    if delta < thresh:
        grid.render_v(v, random_policy, print_value=True)
        break
