import time

from search.base import BaseSearchAgent


class SearchAgent(BaseSearchAgent):
    """ General implementation of a search agent.
    Uses the given search method to search for a solution to the given problem.
    """

    def __init__(self, problem, method, heuristic=None, verbose=False):
        """ Initialize a search agent instance.

        @param problem (searchProblem): A search problem class.
        @param method (func): A search method function.
        @param heuristic (func): A search heuristic function to be used with the search
            method function.
        """
        self.problem = problem
        self.heuristic = heuristic
        if 'heuristic' not in method.__code__.co_varnames:            
            self.searchFunction = lambda x: method(x, verbose=verbose)
        else:
            self.searchFunction = lambda x: method(x, heuristic=heuristic, verbose=verbose)

    def registerInitialState(self, state):
        """ This is the first time that the agent sees the layout of the game board. Here,
        we choose a path to the goal. In this phase, the agent should compute the path to
        the goal and store it in a local variable.
        All of the work is done in this method!

        @param state (gameState): A GameState object.
        """
        if self.searchFunction == None:
            raise Exception("No search method provided for SearchAgent")
        tic = time.time()
        problem = self.problem(state)               # make a new search problem
        self.actions = self.searchFunction(problem) # find a path
        totalCost = problem.getCostOfActions(self.actions)
        toc = time.time()
        print(f"Path found with total cost of {totalCost} in {toc-tic:.6f} seconds")
        if "_expanded" in dir(problem): print(f"Search nodes expanded: {problem._expanded}")

    def getAction(self, state):
        """ Returns the next action in the path chosen earlier (in registerInitialState).
        Return Directions.STOP if there is no further action to take.

        @param state (gameState): A GameState object.
        @return action (Any): The action chosen for the player.
        """
        if "actionIndex" not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return state.getStopAction()    # ????

#