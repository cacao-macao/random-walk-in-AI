from collections import defaultdict
import random

from mdp.base import BaseMDPAgent
from util import flipCoin


class ValueIterationAgent(BaseMDPAgent):
    """
        A ValueIterationAgent takes a Markov decision process (see base.py) on
        initialization and runs value iteration for a given number of iterations using
        the supplied discount factor.
    """

    def __init__(self, mdp, method, discount=0.9):
        """ Initialize a value iteration agent and run the indicated number of iterations.
        The agent acts according to the resulting policy.
        
        @param mdp (MarkovDecisionProcess): A markov decision process class.
        @param method (func): Value iteration method.
        @param discount (float): Discount factor.
        """
        self.mdp_factory = mdp
        self.mpd = None
        self.method = method
        self.discount = discount
        self.values = defaultdict(lambda x: 0)

    def runValueIteration(self, layout, num_iters):
        """
        @param iterations (int): Number of iterations.
        """
        self.mdp = self.mdp_factory(layout)
        self.values = self.method(self.mdp, self.discount, num_iters)

    def value(self, state):
        return self.values[state]

    def q_value(self, state, action):
        q_value = sum((prob * (self.mdp.reward(state, action, next_s) + self.discount*self.values[next_s])
            for next_s, prob in self.mdp.transitions_and_probs(state,action)))
        return q_value

    def policy(self, state):
        act = None
        max_value = float("-inf")
        for a in self.mdp.actions(state):
            q_value = self.q_value(state, a)
            if max_value < q_value:
                max_value = q_value
                act = a
        return act

    def action(self, state, epsi=0.0):
        if flipCoin(epsi):
            return random.choice(self.mdp.actions(state))
        else:
            return self.policy(state)


class RandomAgent(BaseMDPAgent):
    def q_value(self, state, action):
        return 0.0
    def value(self, state):
        return 0.0
    def policy(self, state):
        return random.choice(self.mdp.actions(state))
    def action(self, state):
        return self.policy(state)

#