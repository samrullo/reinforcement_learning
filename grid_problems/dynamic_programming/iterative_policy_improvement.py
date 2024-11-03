from common.grid_world import GridWorld, State, Action
from collections import defaultdict


# we will implement iterative policy improvement
# we go through policy improvement and value function evaluation under new policy
# when policy stops to improve we conclude that we reached an optimal policy
# policy is improved by assigning probabilities to action values greedily.
# I.e. we assign probability of 1 to the action that maximizes value

def one_step_eval(env: GridWorld, v: defaultdict, policy: defaultdict, gamma: float = 0.9):
    for state in env.states():
        if state == env.goal_state:
            v[state] = 0
            continue
        state_value = 0
        action_probs = policy[state]
        for action, action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            reward = env.reward(state, action, next_state)
            state_value += action_prob * (reward + gamma * v[next_state])
        v[state] = state_value
    return v


def evaluate_policy(env: GridWorld, v: defaultdict, policy: defaultdict, gamma: float = 0.9, thresh: float = 0.001):
    while True:
        v_old = v.copy()
        v = one_step_eval(env, v, policy, gamma)
        delta = 0
        for state in v.keys():
            t = abs(v[state] - v_old[state])
            if delta < t:
                delta = t
        if delta < thresh:
            break
    return v


def calc_action_value(env: GridWorld, v: defaultdict, state: State, action: Action, gamma: float = 0.9):
    next_state = env.next_state(state, action)
    reward = env.reward(state, action, next_state)
    return reward + gamma * v[next_state]


def greedy_policy(v: defaultdict, env: GridWorld, gamma: float = 0.9):
    policy = defaultdict(lambda: {action: 0.0 for action in env.actions})
    for state in env.states():
        action_values = {action: calc_action_value(env, v, state, action, gamma) for action in env.actions}
        max_action_value = max(action_values.values())
        max_action = Action.TOP
        for action, action_value in action_values.items():
            if action_value == max_action_value:
                max_action = action
        action_probs = {action: 1.0 if action == max_action else 0.0 for action in env.actions}
        policy[state] = action_probs
    return policy


if __name__ == "__main__":
    env = GridWorld()
    gamma = 0.9
    thresh = 0.001
    pi = defaultdict(lambda: {action: 1 / len(env.actions) for action in env.actions})
    v = defaultdict(lambda: 0)
    cnt = 0

    while True:
        v = evaluate_policy(env, v, pi, gamma, thresh)
        new_pi = greedy_policy(v, env, gamma)

        if new_pi == pi:
            break

        pi = new_pi
        cnt += 1
        print(f"Iteration {cnt} completed")
    env.render_v(v, pi, print_value=True)
