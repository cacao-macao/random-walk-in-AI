from search.base import BaseSearchProblem
from search.search_methods import bfs


class PositionSearchProblem(BaseSearchProblem):
    """ This search problem is used to find paths to a particular point on the pacman board.
    The state space consists of (x,y) positions in a pacman game.
    """

    def __init__(self, gameState, costFn=lambda x: 1, goal=(1,1), warn=True, visualize=True):
        """ Initialize a position search problem.

        @param gameState (gameState): A GameState object (pacman.py).
        @param costFn (func): A function from a search state to a non-negative number.
        @param goal (Tuple(int, int)): A position in the gameState.
        @param warn (bool): If True, print a warning message if this is not a valid
            position search problem.
        @param visualize (bool): If True, visualize a heat map over the states visited
            during solving the search problem.
        """
        self.gameState = gameState.deepCopy()
        self.walls = gameState.getWalls()
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print("Warning: this does not look like a regular search maze")
        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.gameState.getPacmanPosition()

    def isGoalState(self, state):
        isGoal = state == self.goal
        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable
        return isGoal

    def getSuccessors(self, state):
        """ For a given state, this should return a list of triples:
                (successor, action, stepCost)
    
        @param state (Tuple(int, int)): Current pacman position in the game state.
        @return result (List[(state, action, cost)]): A list of successor states, the
            actions they require, and the step cost.
        """
        successors = []
        for action in self.gameState.getAllActions():
            x, y = state
            dx, dy = self.gameState.getPacmanDelta(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))
        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)
        return successors

    def getCostOfActions(self, actions):
        """ Returns the cost of a particular sequence of actions. If those actions include
        an illegal move, return 999999.
        """
        if actions == None: return 999999
        x, y = self.getStartState()
        cost = 0
        for action in actions:
            dx, dy = self.gameState.getPacmanDelta(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost


class CornersProblem(BaseSearchProblem):
    """ This search problem is used to find paths through all four corners of a layout.
    A search state in this problem is a tuple (pacmanPosition, visitedCorners), where:
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      visitedCorners: a set, keeping track of the already visited corners
    """

    def __init__(self, gameState, costFn=lambda x: 1, warn=True):
        """ Initialize a corners problem.

        @param gameState (gameState): A GameState object (pacman.py).
        @param costFn (func): A function from a search state (tuple) to a non-negative number.
        @param warn (bool): If True, print a warning message if this is not a valid
            position search problem.
        """
        self.gameState = gameState.deepCopy()
        self.walls = gameState.getWalls()
        self.costFn = costFn
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if warn and not gameState.hasFood(*corner):
                print("Warning: no food in corner " + str(corner))
        self._expanded = 0 # Number of search nodes expanded

    def getStartState(self):
        return (self.gameState.getPacmanPosition(), tuple())


    def isGoalState(self, state):
        _, visitedCorners = state
        return len(set(self.corners) - set(visitedCorners)) == 0

    def getSuccessors(self, state):
        """ For a given state, this should return a list of triples:
                (successor, action, stepCost)
    
        @param state (Tuple(int, int)): Current pacman position in the game state.
        @return result (List[(state, action, cost)]): A list of successor states, the
            actions they require, and the step cost.
        """
        successors = []
        for action in self.gameState.getAllActions():
            pos, visitedCorners = state
            x, y = pos
            dx, dy = self.gameState.getPacmanDelta(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextPos = (nextx, nexty)
                if nextPos in self.corners and nextPos not in visitedCorners:
                    nextState = (nextPos, visitedCorners + (nextPos,))
                else:
                    nextState = (nextPos, visitedCorners)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))
        self._expanded += 1
        return successors

    def getCostOfActions(self, actions):
        """ Returns the cost of a particular sequence of actions. If those actions include
        an illegal move, return 999999.
        """
        if actions == None: return 999999
        (x,y), visited = self.getStartState()
        cost = 0
        for action in actions:
            dx, dy = self.gameState.getPacmanDelta(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            nextPos = (x, y)
            if nextPos in self.corners and nextPos not in visited:
                visited + (nextPos,)
            cost += self.costFn((nextPos, visited))
        return cost


class FoodSearchProblem(BaseSearchProblem):
    """ This search problem is used to find a path that collects all of the food (dots) in
    a Pacman game.
    A search state in this problem is a tuple (pacmanPosition, foodGrid) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """

    def __init__(self, gameState, costFn=lambda x: 1):
        """ Initialize a food search problem.

        @param gameState (gameState): A GameState object (pacman.py).
        @param costFn (func): A function from a search state (tuple) to a non-negative number.
        """
        self.gameState = gameState.deepCopy()
        self.walls = gameState.getWalls()
        self.costFn = costFn
        self.goal = None
        self._expanded = 0      # Number of search nodes expanded
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return (self.gameState.getPacmanPosition(), self.gameState.getFood())

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        """ For a given state, this should return a list of triples:
                (successor, action, stepCost)
        """
        successors = []
        for action in self.gameState.getAllActions():
            (x,y), *_ = state
            dx, dy = self.gameState.getPacmanDelta(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                nextState = ((nextx, nexty), nextFood)
                cost = self.costFn(nextState)
                successors.append((nextState, action, cost))
        self._expanded += 1
        return successors

    def getCostOfActions(self, actions):
        """ Returns the cost of a particular sequence of actions. If those actions include
        an illegal move, return 999999.
        """
        if actions == None: return 999999
        (x,y), food = self.getStartState()
        cost = 0
        for action in actions:
            dx, dy = self.gameState.getPacmanDelta(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            food[x][y] = False
            nextState = ((x, y), food)
            cost += self.costFn(nextState)
        return cost


#----------------------- Heuristic functions for informed search ------------------------#

def manhattanDistance(position, goal):
    """ Returns the Manhattan distance between points position and goal. """
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])

def euclideanDistance(position, goal):
    """ Returns the Euclidean distance between points position and goal. """
    return ((position[0]-goal[0])**2 + (position[1]-goal[1])**2) ** 0.5

def mazeDistance(point1, point2, gameState):
    """ Returns the maze distance between any two points, using breadth first search.
    The gameState can be any game state -- Pacman's position in that state is ignored.

    @param point1 (Tuple(int, int)): (x, y) coordinates of the first point.
    @param point2 (Tuple(int, int)): (x, y) coordinates of the seconde point.
    @param gameState (gameState): A GameState object (pacman.py).
    @return len (int): Minimal number of moves needed to get from point1 to point2.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], "point1 is a wall: " + str(point1)
    assert not walls[x2][y2], "point2 is a wall: " + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(bfs(prob))

#




# def cornersHeuristic(state, problem):
#     """
#     A heuristic for the CornersProblem that you defined.

#       state:   The current search state
#                (a data structure you chose in your search problem)

#       problem: The CornersProblem instance for this layout.

#     This function should always return a number that is a lower bound on the
#     shortest path from the state to a goal of the problem; i.e.  it should be
#     admissible (as well as consistent).
#     """
#     corners = problem.corners # These are the corner coordinates
#     walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

#     "*** YOUR CODE HERE ***"
#     # return 0 # Default to trivial solution
#     pos, visitedCorners = state
#     remainingCorners = set(corners) - set(visitedCorners)
#     return max([mazeDistance(pos, c, problem.gameState) for c in remainingCorners] + [0])
#
# def foodHeuristic(state, problem):
#     """
#     Your heuristic for the FoodSearchProblem goes here.

#     This heuristic must be consistent to ensure correctness.  First, try to come
#     up with an admissible heuristic; almost all admissible heuristics will be
#     consistent as well.

#     If using A* ever finds a solution that is worse uniform cost search finds,
#     your heuristic is *not* consistent, and probably not admissible!  On the
#     other hand, inadmissible or inconsistent heuristics may find optimal
#     solutions, so be careful.

#     The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
#     (see game.py) of either True or False. You can call foodGrid.asList() to get
#     a list of food coordinates instead.

#     If you want access to info like walls, capsules, etc., you can query the
#     problem.  For example, problem.walls gives you a Grid of where the walls
#     are.

#     If you want to *store* information to be reused in other calls to the
#     heuristic, there is a dictionary called problem.heuristicInfo that you can
#     use. For example, if you only want to count the walls once and store that
#     value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
#     Subsequent calls to this heuristic can access
#     problem.heuristicInfo['wallCount']
#     """
#     position, foodGrid = state
#     "*** YOUR CODE HERE ***"
#     # return 0
#     return max([mazeDistance(position, (c,r), problem.startingGameState)
#                 for r in range(foodGrid.height)
#                 for c in range(foodGrid.width) if foodGrid[c][r]] + [0])
#
#