class BaseSearchProblem:
    """ An abstract class implementation of a search problem.
    A search problem defines the state space, start state, goal test, successor
    function and cost function. 
    This class outlines the structure of a search problem. Concrete classes must implement
    the given methods.
    """

    def getStartState(self):
        """ Returns the start state for the search problem. """
        raise NotImplementedError("This method must be implemented by the subclass")

    def isGoalState(self, state):
        """ Returns True if and only if the state is a valid goal state. """
        raise NotImplementedError("This method must be implemented by the subclass")

    def getSuccessors(self, state):
        """ For a given state, this method should return a list of triples:
            (successor, action, stepCost),
        where 'successor' is a successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental cost of expanding to that
        successor.
        """
        raise NotImplementedError("This method must be implemented by the subclass")

    def getCostOfActions(self, actions):
        """ This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        raise NotImplementedError("This method must be implemented by the subclass")


class BaseSearchAgent:
    """ An abstract class implementation of a search agent.
    The search agent plans through a search problem using a supplied search algorithm.
    This class outlines the structure of a search agent. Concrete classes must the
    given methods.
    """

    def __init__(self, problem, method, heuristic):
        """ The agent takes a search problem and uses the provided search method and
        heuristic to plan through the problem.
        """
        raise NotImplementedError("This method must be implemented by the subclass")

    def registerInitialState(self, state):
        """ This is the first time that the agent sees the layout of the search problem.
        In this phase, the agent should plan the path to the goal and store it in a local
        variable.
        All of the work is done in this method!
        """
        raise NotImplementedError("This method must be implemented by the subclass")

    def getAction(self, state):
        """ The Agent will receive a state from a search problem and must return a legal
        action.
        """
        raise NotImplementedError("This method must be implemented by the subclass")

#
