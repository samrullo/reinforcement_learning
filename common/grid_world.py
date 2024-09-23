import numpy as np
from enum import Enum, IntEnum
from collections import namedtuple, defaultdict
from common.gridworld_render import Renderer


class Action(IntEnum):
    TOP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


State = namedtuple("State", "y x")

from typing import List


class GridWorld:
    def __init__(self, reward_map: np.array = None, wall_states: List[State] = None, apple_states: List[State] = None,
                 bomb_states: List[State] = None, goal_state: State = None, goal_state_reward: float = None):
        self.actions = Action
        if goal_state is None:
            self.goal_state = State(0,3)
        else:
            self.goal_state = goal_state

        if apple_states is None:
            self.apple_states = [State(0, 3)]
        else:
            self.apple_states = apple_states

        if bomb_states is None:
            self.bomb_states = [State(1, 3)]
        else:
            self.bomb_states = bomb_states

        if reward_map is None:
            self.reward_map = np.zeros((3, 4))
        else:
            self.reward_map = reward_map

        # apple
        for state in self.apple_states:
            self.reward_map[state] = 1
        # bomb
        for state in self.bomb_states:
            self.reward_map[state] = -1

        if goal_state_reward is None:
            self.reward_map[self.goal_state] = len(self.apple_states)
        else:
            self.reward_map[self.goal_state] = goal_state_reward

        # A move to the right is changing (y,x) cell coordinates by (0,1)
        # A move to the left is changing (y,x) cell coordinates by (0,-1)
        # A move to the top is changing (y,x) cell coordinates by (-1,0)
        # A move to the bottom is changing (y,x) cell coordinates by (1,0)
        self.action_moves = {Action.RIGHT: (0, 1), Action.LEFT: (0, -1), Action.TOP: (-1, 0), Action.DOWN: (1, 0)}

        if wall_states is None:
            self.wall_states = [State(1, 1)]
        else:
            self.wall_states = wall_states

        self.start_state = State(0, 0)
        self.agent_state = self.start_state

        if goal_state is None:
            self.goal_state = State(0, self.width - 1)
        else:
            self.goal_state = goal_state

    @property
    def height(self):
        return self.reward_map.shape[0]

    @property
    def width(self):
        return self.reward_map.shape[1]

    def states(self):
        for y in range(self.height):
            for x in range(self.width):
                yield State(y, x)

    def reward(self, state: State, action: Action, next_state: State):
        return self.reward_map[next_state]

    def next_state(self, state: State, action: Action):
        y_move = self.action_moves[action][0]
        x_move = self.action_moves[action][1]
        thenext_state = State(state.y + y_move, state.x + x_move)

        # if breaching edges
        if thenext_state.y >= self.height or thenext_state.y < 0 or thenext_state.x >= self.width or thenext_state.x < 0 or thenext_state in self.wall_states:
            thenext_state = state
        return thenext_state

    def step(self, action: Action):
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        if next_state == self.goal_state:
            done = True
        else:
            done = False
        self.agent_state = next_state
        return next_state, reward, done

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = Renderer(self.reward_map, self.goal_state, self.wall_states)
        renderer.render_v(v, policy, print_value=print_value)

    def render_q(self, q=None, show_greedy_policy=True):
        renderer = Renderer(self.reward_map, self.goal_state, self.wall_states)
        renderer.render_q(q, show_greedy_policy=show_greedy_policy)


if __name__ == "__main__":
    grid = GridWorld()
    next_state, reward, done = grid.step(Action.RIGHT)
    print(f"after one move to right, next_state : {next_state}, reward : {reward}, done:{done}")
    next_state, reward, done = grid.step(Action.RIGHT)
    print(f"after two moves to right, next_state : {next_state}, reward : {reward}, done:{done}")
    next_state, reward, done = grid.step(Action.RIGHT)
    print(f"after three moves to right, next_state : {next_state}, reward : {reward}, done:{done}")
    from common.gridworld_render import Renderer

    renderer = Renderer(grid.reward_map, grid.goal_state, grid.wall_state)
    v = defaultdict(lambda: 0)
    v[State(0, 2)] = 0.9
    v[State(0, 1)] = 0.9
    random_actions = {action: 1 / len(Action) for action in Action}
    policy = defaultdict(lambda: random_actions)
    renderer.render_v(v, policy, print_value=True)
