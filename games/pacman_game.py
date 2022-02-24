import random

from games.base import BaseGameState


class PacmanGameState(BaseGameState):
    """ PacmanGameStates represents an abstraction of the pacman game.
    The class implements the methods of the abstract base class.
    """

    def __init__(self, gameState, to_move=0):
        """ Initialize a pacman game state instance.
        
        @param gameState (gameState): The state of the game of pacman.
        @param to_move (int): The id of the player that is to make a move next.
        """
        self._state = gameState
        self._to_move = to_move

    def actions(self):
        """ Returns a list of legal actions for the player which is to move next. """
        agentID = self._to_move
        return self._state.getLegalActions(agentID)

    def to_move(self):
        """ Return the id of the player whose turn it is to play next. """
        return self._to_move

    def is_terminal(self):
        """ Returns true if the game state is a terminal state."""
        return self._state.isWin() or self._state.isLose()

    def agent_id(self):
        """ Returns the ID of the player controlled by the agent. """
        return 0

    def score(self):
        """ Return the score of the game state. """
        return self._state.getScore()

    def result(self, action):
        """ Returns the successor game state after a player takes an action. """
        playerID = self._to_move
        new_state = self._state.generateSuccessor(playerID, action)
        return PacmanGameState(new_state, (playerID+1) % self._state.getNumAgents())


#------------------------------------ Utility funcs -------------------------------------#
def scoreEvaluationFunction(gameState):
    """ This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
    """
    return gameState.score()

def betterEvaluationFunction(gameState):
    currentGameState = gameState._state
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    foodPositions = ([(c, r) for r in range(food.height)
                            for c in range(food.width) if food[c][r]])
    minFoodDist = min((manhattanDistance(position, food) for food in foodPositions))
    ghostDistance = min([manhattanDistance(position, ghost.getPosition()) for ghost in ghostStates] + [2])
    randomBoost = flipCoin(0.35) * 35
    score = currentGameState.getScore() - minFoodDist - 10 ** (4 - ghostDistance) + randomBoost
    return score

def manhattanDistance(position, goal):
    """ Returns the Manhattan distance between points position and goal. """
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

def flipCoin(p):
    r = random.random()
    return r < p

#