class BaseGameState:
    """ An abstract class implementation of a game state.
    A game state defines the legal actions a player can take, keeps track of which player
    is to move, tests if the current state is a terminal state for the game, and produces
    successor game states given an action taken by the player.
    This class outlines the structure of a game state. Concrete classes must implement
    the given methods.
    """

    def actions(self):
        """ Returns a list of legal actions for the player which is to move next. """
        raise NotImplementedError("This method must be implemented by the subclass")

    def to_move(self):
        """ Return the id of the player whose turn it is to play next. """
        raise NotImplementedError("This method must be implemented by the subclass")

    def is_terminal(self):
        """ Returns true if the game state is a terminal state."""
        raise NotImplementedError("This method must be implemented by the subclass")

    def agent_id(self):
        """ Returns the ID of the player controlled by the agent. """
        raise NotImplementedError("This method must be implemented by the subclass")

    def score(self):
        """ Return the score of the game state. """
        raise NotImplementedError("This method must be implemented by the subclass")

    def result(self, action):
        """ Returns the successor game state after a player takes an action. """
        raise NotImplementedError("This method must be implemented by the subclass")

#