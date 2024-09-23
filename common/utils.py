import numpy as np
from common.grid_world import State, Action


def greedy_probs(Q, state: State, epsilon: float):
    qs = [Q[state, a] for a in Action]
    max_action = np.argmax(qs)
    action_probs = {a: epsilon / len(Action) for a in Action}
    action_probs[max_action] += 1 - epsilon
    return action_probs
