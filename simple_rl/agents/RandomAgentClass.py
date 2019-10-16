''' RandomAgentClass.py: Class for a randomly acting RL Agent '''

# Python imports.
import random

# Other imports
from simple_rl.agents.AgentClass import Agent

class RandomAgent(Agent):
    ''' Class for a random decision maker. '''

    def __init__(self, actions, name=""):
        name = "Random" if name is "" else name
        Agent.__init__(self, name=name, actions=actions)

    def act(self, state, reward):
        return random.choice(self.actions)

class RandomContAgent(Agent):
    ''' Class for a random decision maker. '''

    def __init__(self, action_dim, action_bound, name=""):
        name = "Random" if name is "" else name
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.name = name
        Agent.__init__(self, name=name, actions=[])

    def act(self, state, reward, learning=True):
        actions = np.zeros(self.action_dim)
        for i in range(len(actions)):
            actions[i] = random.uniform(self.action_bound[0][i], self.action_bound[1][i])
        return actions
