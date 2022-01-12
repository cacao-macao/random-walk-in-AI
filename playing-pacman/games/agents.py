class AdversarialAgent:
    """ General implementation of an adversarial search agent.
    Uses the given game search method to search for the best move given the current game
    state. The search explores the game tree up to e given depth and uses the utility
    function to evaluate non-terminal leaf states.
    """

    def __init__(self, game, method, utility, depth):
        """ Initialize an adversarial search agent instance.

        @param game (gameState): A game state class.
        @param method (func): A game search method function.
        @param utility (func): An evaluation function used to score the leaves of the
            search tree in case they are not terminal states.
        @param depth (int): Maximum depth of the search tree.
        """
        self._game = game
        self._method = method
        self._utility = utility
        self._depth = depth

    def getAction(self, state):
        """ Returns the next action chosen using the game search method.

        @param state (gameState): A GameState object.
        @return action (Any): The action chosen for the player.
        """
        gameState = self._game(state)
        return self._method(gameState, self._depth, self._utility)

#