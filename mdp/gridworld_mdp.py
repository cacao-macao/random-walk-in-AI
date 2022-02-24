from mdp.base import BaseMarkovDecisionProcess

class GridWorldMDP(BaseMarkovDecisionProcess):
    """ A Markov Decision Process representation of the GridWorld problem. """

    def __init__(self, gridWorldLayout):
        self.gridworld = gridWorldLayout

    def start_state(self):
        """ Returns the start state for the markov decision process. """
        return self.gridworld.getStartState()

    def states(self):
        """ Returns a list of all states of the markov decision process. """
        return self.gridworld.getStates()

    def actions(self, state):
        """ Returns a list of legal actions from the given state. """
        return self.gridworld.getPossibleActions(state)

    def transitions_and_probs(self, state, action):
        """ Returns a list of (next_state, prob) pairs representing the states reachable
        from `state` by taking `action` along with their transition probabilities.
        """
        return self.gridworld.getTransitionStatesAndProbs(state, action)

    def reward(self, state, action, next_state):
        """ Returns the reward for the (state, action, next_state) transition. """
        return self.gridworld.getReward(state, action, next_state)

    def is_terminal(self, state):
        """ Returns true if the state is a terminal state. """
        return self.gridworld.isTerminal(state)

#