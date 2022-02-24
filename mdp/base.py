class BaseMarkovDecisionProcess:
    """ An abstract class implementation of a markov decision process.
    A markov decision process defines the start state, state space, action space, reward
    space, transition function and terminal test.
    This class outlines the structure of an MPD. Concrete classes must implement the given
    methods.
    """

    def start_state(self):
        """ Returns the start state for the markov decision process. """
        raise NotImplementedError("This method must be implemented by the subclass")

    def states(self):
        """ Returns a list of all states of the markov decision process. """
        raise NotImplementedError("This method must be implemented by the subclass")

    def actions(self, state):
        """ Returns a list of legal actions from the given state. """
        raise NotImplementedError("This method must be implemented by the subclass")

    def transitions_and_probs(self, state, action):
        """ Returns a list of (next_state, prob) pairs representing the states reachable
        from `state` by taking `action` along with their transition probabilities.
        """
        raise NotImplementedError("This method must be implemented by the subclass")

    def reward(self, state, action, next_state):
        """ Returns the reward for the (state, action, next_state) transition. """
        raise NotImplementedError("This method must be implemented by the subclass")

    def is_terminal(self, state):
        """ Returns true if the state is a terminal state. """
        raise NotImplementedError("This method must be implemented by the subclass")


class BaseMDPAgent:
    """
      Abstract agent which assigns values to (state,action)
      Q-Values for an environment. As well as a value to a
      state and a policy given respectively by,

      V(s) = max_{a in actions} Q(s,a)
      policy(s) = arg_max_{a in actions} Q(s,a)

      Both ValueIterationAgent and QLearningAgent inherit
      from this agent. While a ValueIterationAgent has
      a model of the environment via a MarkovDecisionProcess
      (see mdp.py) that is used to estimate Q-Values before
      ever actually acting, the QLearningAgent estimates
      Q-Values while acting in the environment.
    """

    def q_value(self, state, action):
        """ Returns the value of the (state, action) pair. """
        raise NotImplementedError("This method must be implemented by the subclass")

    def value(self, state):
        """ Returns the value of the state under the optimal policy. """
        raise NotImplementedError("This method must be implemented by the subclass")

    def action(self, state):
        """ Returns the action to be taken in this state. """
        raise NotImplementedError("This method must be implemented by the subclass")

    def policy(self, state):
        """ Returns the best action to take in the state. Note that because we might want
        to explore, this might not coincide with `action()`.
        """
        raise NotImplementedError("This method must be implemented by the subclass")

#