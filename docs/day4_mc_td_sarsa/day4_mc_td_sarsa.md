# Monte Carlo method
In the previous chapters we were able to find ```Optimal Policy``` without moving Agent around in the Environment. That's because we knew the whole complete Environment model, all possible states, all state transitions, all rewards possible. 

However in reality it is impossible to know Environment model beforehand. Even if it is possible, the computation is impossible due to sheer volume. Hence, Reinforcement Learning primarily deals with problems where Environment model is unknown. Under unknown Environment model Agent tries to find Optimal Policy. For that Agent will have to act in the environment and learn from experience.

So how can the Agent approximate ```state value```s without knowing probability of state transitions and without knowing all possible states.
One way is known as Monte Carlo method. Monte Carlo methods is all about sampling. If we return to our Grid Problem, we could let our Agent start from a certain ```state``` and play till the episode ends. While the Agent plays it will take actions, transition to new states and recieve rewards. Remember the definition of ```state value```. It was the expected value of ```Return``` when Agent starts from that ```State```.

$$
V_{\pi}(S_t) = E_{\pi}[G_t|S_t]
$$

And what was the definition of ```Return```? It was sum of exponentially decaying ```Reward```s

$$
G_t = R_t + \gamma * R_{t+1} + \gamma^2 * R_{t+2}+...
$$

We could approximate the ```Return``` that the Agent earned in one episode as the exponentially decaying ```Rewards``` that the Agent recieved during an episode. Then we could run multiple episodes, calculate Returns for each episode and approximate expected value of Return as the average of all Returns that the Agent earned across episodes. This is how we can approximate the ```Value``` of a single ```State```.

We could use this method and make the Agent start from different ```State```s and calculate their ```Value```s using above sampling methodology. But it won't work in reality. Because in the game, you usually start from predefined small number of ```State```s. You can not make the Agent start from every possible state. Moreover, you don't know all possible states beforehand.

So how do we calculate ```Values``` for all other ```States```.

Below is how

1. Agent collects samples consisting of (```State```,```Action```, ```Reward```) and keeps on adding them into memory. So memory will look like below

$$
(S_t, A_t, R_t), (S_{t+1}, A_{t+1}, R_{t+1}), (S_{t+2}, A_{t+2}, R_{t+2}),...
$$

2. Let's assume a very simple scenario where there were three states A,B,C. Agent played an episode and memory looked like below

$$
(A,action, R_1), (B, action, R_2), (C, action, R_3),END
$$

In above simple example the Agent started from state A, took an action, recieved reward R1, transitioned to state B, took an action, recieved reward R2, transitioned to state C took an action, recieved reward R3 and reached the end of the episode. We can calculate ```Returns``` of each state A, B, C in the following manner

$$
G_A = R_1 + \gamma * R_2 + \gamma^2 * R_3
$$

$$
G_B = R_2 + \gamma * R_3
$$

$$
G_C = R_3
$$

We could rearrange above ```Returns``` and express the ```Returns``` of each state in terms of next ```State``` that Agent transitions to.

$$
G_C = R_3
$$

$$
G_B = R_2 + \gamma * G_C
$$

$$
G_A = R_1 + \gamma * G_B
$$

Using above rearrangement, we can save compute resources and still calculate the ```Return``` of each visited ```State``` in the episode.

Now that we know how to calculate ```Returns``` for all ```State```s visited in an episode,
we can run multiple episodes, calculate ```Returns``` for ```States``` in each episode and 
calculate ```Value``` of ```State```s by taking average of ```Returns``` over episodes.


The pseudocode could look something like below

```python
class Agent:
    def __init__(self):
        self.memory = []
        self.actions = [a for a in Action]
        self.pi = defaultdict(lambda : {a : 1/len(self.actions) for a in self.actions})        
        self.gamma = 0.9
        self.agent_returns = defaultdict(lambda : [])
    
    def get_action(self, state):
        if random.rand() < self.epsilon:
            return random.sample(self.actions)
        else:
            return self.pi[state]
    
    def add_to_memory(self,state,action,reward):
        self.memory.append((state,action,reward))
    
    def reset(self):
        self.memory=[]        
    
    def calculate_returns(self):
        G = 0
        # iteratee over memory in reverse order to save compute time when calculating Return for a state
        for data in reversed(self.memory):
            state,action,reward = data

            # calculate Return of state
            G = reward + self.gamma * G

            self.agent_returns[state].append(G)

episodes = 1000
V = defaultdict(lambda : 0)

for e in range(episodes):
    state = env.reset()
    done = False    
    cnt = {}
    while not done:
        action = agent.get_action(state)

        next_state, reward, done = env.step(state,action)

        agent.add_to_memory((state,action,reward))
    
    # at this point episode ended so we can calculate Returns
    agent.calculate_returns()

# we ran multiple episodes and sampled multiple Returns for States
# now we can approximate Value of States taking average of Returns
for state, agent_returns in agent.agent_returns.items():
    V[state] = np.mean(agent_returns)
```

There is one problem with above approach. We are storing calculateed ```Return```s for each ```State``` over episodes
in a list. We are wasting memory.

There is a better approach. We could continuously update ```State Value```s as we calculate ```Return```s in each episode 
without storing ```Return```s in list and fill up computer memory. Although we understand that for small problems like GridWorld there is more than enough memory in computer,
once we work on a more complex problems we could easily run out of memory.

So how can we calculate ```Value```s continuously using ```Returns``` computed in each episode.

Using below formula

$$
V_{n-1} = (G_1 + G_2 + G_3 + ... + G_{n-1})/(n-1)
$$

$$
(n-1) * V_{n-1} = G_1 + G_2 + G_3 + ... + G_{n-1}
$$

$$
V_n = (G_1 + G_2 + G_3 + ... + G_n)/n
$$

$$
V_n = ((n-1)*V_{n-1}  + G_n)/n = (n * V_{n-1} - V_{n-1} + G_n)/n
$$

$$
V_n = V_{n-1} + 1/n * (G_n - V_{n-1})
$$

So we could rewrite our pseudocode above as below


```python
class Agent:
    def __init__(self):
        self.memory = []
        self.actions = [a for a in Action]
        self.pi = defaultdict(lambda : {a : 1/len(self.actions) for a in self.actions})        
        self.gamma = 0.9
        self.V = defaultdict(lambda : 0)
        self.cnt = defaultdict(lambda : 0)
    
    def get_action(self, state):
        if random.rand() < self.epsilon:
            return random.sample(self.actions)
        else:
            return self.pi[state]
    
    def add_to_memory(self,state,action,reward):
        self.memory.append((state,action,reward))
    
    def reset(self):
        self.memory=[]        
    
    def eval(self):
        G = 0
        # iteratee over memory in reverse order to save compute time when calculating Return for a state
        for data in reversed(self.memory):
            state,action,reward = data

            # calculate Return of state
            G = reward + self.gamma * G

            self.cnt[state]+=1
            
            self.V += (G - self.V) / self.cnt[state]

episodes = 1000

for e in range(episodes):
    agent.reset()
    state = env.reset()
    done = False    
    cnt = {}
    while not done:
        action = agent.get_action(state)

        next_state, reward, done = env.step(state,action)

        agent.add_to_memory((state,action,reward))
    
    # at this point episode ended so we can calculate Returns and update State Values simultaneously
    agent.eval()

V = agent.V
```

