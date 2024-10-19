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

Now that we derived our Bellman Equation let's go ahead and use it to calculate values of each state for our simple 2 cell problem. Let's consider the policy where the agent randomly chooses right of left at each state.

# Calculating Values using Bellman Equation
When you picked up your pen or opened your python console to embark on the journey of calculating values, you will soon stumble upon the barrier. You think "Wait a minute. In the formula, the value of current state depends on the values of next states. How do I come up with the values of next states?"

What do you think? How? Well, it is simple, just assume at the start that values of all states are zero. But that's not the end of the story right?

Yes, you are right. This method estimating a value using yet other estimations is called bootstrapping. And to get close to the true value, you will have to run this process of updating estimations using estimations multiple times.

Here's how the algorithm to calculate state value using Bellman Equation works

1. You initialize state values of all states as zero
2. You calculate values of each state using Bellman Equation
3. Now that you have updated values for each state, you take absolute difference between new values and older values. And you calculate threshold as the maximum of absolute differences.
4. You continue steps 2 to 3 until the threshold is less than some small value epsilon (may be 0.0001). When your threshold is very very small it is an indication that you got close to true values and hence updates are getting very small.

For our 2 cell problem, or grid problems in general, each state can be represented as coordinates.
For a 2D grid you can represent each state as a tuple with 2 elements. For instance (0,0) may represent topmost lef cell, then (0,1) may represent the cell to the right etc..

To simplify our job, we are going to assume state transition as deterministic. This means when agent chooses action *a*, the agent will transition to a concrete next state *s'* and we can represent determination of next state as a function

$$
s' = f(s,a)
$$

We can represent action probabilities as dictionaries.
For instance random policy can be represented as python dictionary.
Here cell L1 is represented as tuple (0,0) and cell L2 is represented as tuple (0,1). And action left is representedy by integer 0 and action right is represented by integer 1

```python
pi = {
      (0,0): {0:0.5, 1:0.5},
      (0,1): {0:0.5, 1:0.5}
    }
```

With above policy we can easily access action probabilities for each state.

We can also represent values of states as a dictionary

```python
V = {(0,0):0, (0,1):0}
```