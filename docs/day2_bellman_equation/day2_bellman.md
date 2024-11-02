# Bellman Equation
In previous chapter we were able to list all possible policies for a 2 cell problem and directly calculate values for each cell under each policy
and was able to easily identify the optimal policy.
You now know that ability to calculate *value of states* under a policy is essential to finding the optimal policy.
In most problems, we can not directly calculate value of a state. It becomes intractable once the number of states and the possible numbere of actions increase slightly.

Hence, we need another way of estimating *state values*.

Bellman equation allows us to do just that.


We start with the definition of *state value*. It was expected value of *Return* when the agent starts from that state.
Why expected value? Because remember *Return* is probabilistic, which means even if the agent starts from the same state, its Return may result in different values at each attempt.
So ideally we want an average *Return* over high number of attempts.

Let's write down *state value* initial equation

$$
V_{\pi}(s) = E_{\pi}[G_t | S_t = s]
$$

We can rewrite above as 

$$
V_{\pi}(s) = E_{\pi}[(R_t + \gamma * G_{t+1}) | S_t = s]
$$

Rt stands for reward earned at time t when agent starts from state *s*.

Let's think how can we calculate expected value of reward at time t when agent starts from state *s*. When at state *s*, agent may choose action *a* from action space and transition to next state *s'*. Reward is a function of all of these, meaning there can be different rewards when agent chooses particular action *a* and transition to a particular next state *s'*

The way agent chooses action and the way it transitions to next state are all probabilistic and they are denoted with below notations.

Probability of choosing action *a* when at state *s* is known as agent policy and denoted as

$$
\pi(a|s)
$$

Probability of transition to next state *s'* when agent starts from state *s* and chooses action *a* is denoted as

$$
p(s'|s,a)
$$

So now we can rewrite our *state value* formula as

$$
V_{\pi}(s) = \Sigma_a \pi(a|s) \Sigma_{s'}p(s'|s,a) \left[ r(s,a,s') + \gamma * E_{\pi}[G_{t+1} | S_{t+1} = s']\right]
$$

You can replace expected value of Return at t+1 with value of state *s'*

$$
V_{\pi}(s) = \Sigma_a \pi(a|s) \Sigma_{s'}p(s'|s,a) \left[ r(s,a,s') + \gamma * V_{\pi}(s')\right]
$$

And that's it, we just derived Bellman Equation. Bellman Equation shows relationship between the value of current state and the value of all next states. To use Bellman Equation we need *Environment Model*. Environment model in Reinforcement Learning is defined by probability distribution of how agent chooses action when at any state *s* and how the environment transitions to next state *s'* when agent starts from state *s* and chooses action *a*

Now that we derived our Bellman Equation let's go ahead and use it to calculate values of each state for our simple 2 cell problem. Let's consider the policy where the agent randomly chooses right or left at each state.

# Calculating Values using Bellman Equation
When you picked up your pen or opened your python console to embark on the journey of calculating values, you will soon stumble upon the barrier. You think "Wait a minute. In the formula, the value of current state depends on the values of next states. How do I come up with the values of next states?"

What do you think? How? Well, it is simple, just assume at the start that values of all states are zero. But that's not the end of the story right?

Yes, you are right. This method estimating a value using yet other estimations is called bootstrapping. And to get close to the true value, you will have to run this process of updating estimations using estimations multiple times.

Here's how the algorithm to calculate state value using Bellman Equation works

1. You initialize state values of all states as zero
2. You calculate values of each state using Bellman Equation
3. Now that you have updated values for each state, you take absolute difference between new values and older values. And you calculate *threshold* as the maximum of absolute differences.
4. You iterate steps 2 to 3 until the *threshold* is less than some small value epsilon (may be 0.0001). When your threshold is very very small it is an indication that you got close to true values and hence updates are getting very small.

For our 2 cell problem, or grid problems in general, each state can be represented as coordinates.
For a 2D grid you can represent each state as a tuple with 2 elements. For instance (0,0) may represent topmost lef cell, then (0,1) may represent the cell to the right etc..

To simplify our job, we are going to assume state transition as deterministic. This means when agent chooses action *a*, the agent will transition to a concrete next state *s'* and we can represent determination of next state as a function

$$
s' = f(s,a)
$$

We can represent action probabilities as dictionaries.
For instance random policy can be represented as python dictionary.
Here cell L1 is represented as tuple (0,0) and cell L2 is represented as tuple (0,1). And action left is representedy by integer 0 and action right is represented by integer 1

Let's write code to find *Values* of each state under random policy in this simple two cell problem.

We define *actions* as *IntEnum*, this way we can write *Action.LEFT* instead of 0, or *Action.RIGHT* instead of 1.

We also define class *GridEnv* to represent our two cell environment. This environment will implement methods
- To get all states
- To find next state when agent starts from a *state* and takes an *action*
- To get the reward that agent receives when it starts from a *state*, takes an *action* and transitions to *next_state*

```python
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

        # initialize all states. In our case we will have (0,0) to represent L1 cell and (0,1) to represent L2 cell
        states = []
        for h in range(self.height):
            for w in range(self.width):
                states.append((h, w))
        self.states = states

        self.actions = [a for a in Action]

        # what rewards the agent will receive when it visits one of two cells. When it visits cell L2 it recieves reward of +1, when it visits cell L1 it recieves reward of 0
        self.reward_map = np.zeros((self.height, self.width))
        self.reward_map[0, 1] = 1

    def state_transition(self, state, action: Action):
        if action == Action.LEFT:
            next_state = (state[0], state[1] - 1)
        else:
            next_state = (state[0], state[1] + 1)

        x, y = next_state
        # agent can not get outside of its two cell world. When it attempts to do so, it remains in the same cell where it started
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

```