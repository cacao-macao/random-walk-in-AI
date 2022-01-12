from math import sqrt

from envs.blocks.game import SlidingBlocksBoard
from search.base import BaseSearchProblem
from search.search_methods import bfs, ids, astar


class BlocksSearchProblem(BaseSearchProblem):
    """ Implementation of a search problem for the sliding blocks puzzle.

    Each state is represented by tuple of numbers as ordered on the SlidingBlocksBoard.
    """
    def __init__(self, puzzle, costFn=lambda x: 1, goal=-1):
        """ Initialize a sliding blocks search problem.

        @param puzzle (gameState): A sliding blocks board object.
        @param costFn (func): A function from a search state to a non-negative number.
        @param goal (int): A position in the gameState.
        """
        self.puzzle = puzzle
        self.goal = puzzle.goal
        self._startState = (puzzle.numbers, puzzle._rows, puzzle._cols)
        self.costFn = costFn
        self._expanded = 0

    def getStartState(self):
        return self._startState

    def isGoalState(self, state):
        numbers, _, _ = state
        return numbers == self.puzzle._goalState

    def getSuccessors(self, state):
        """ For a given state, this should return a list of triples:
                (successor, action, stepCost)
    
        @param state (gameState): Current state of the game board.
        @return result (List[(state, action, cost)]): A list of successor states, the
            actions they require, and the step cost.
        """
        self._expanded += 1
        numbers, rows, cols = state
        self.puzzle.numbers = numbers
        board = self.puzzle
        successors = []
        for a in board.legalMoves():
            next_state = board.result(a).numbers
            successors.append(((next_state, rows, cols), a, self.costFn(next_state)))
        return successors

    def getCostOfActions(self, actions, heuristic=None):
        """ Returns the cost of a particular sequence of actions. If those actions include
        an illegal move, return 999999.
        """
        if actions == None: return 999999
        start, _, _ = self._startState
        self.puzzle.numbers = start
        board = self.puzzle
        cost = 0
        for action in actions:
            if action not in board.legalMoves(): return 999999
            board = board.result(action)
            cost += self.costFn(board.numbers)
        return cost


#----------------------- Heuristic functions for informed search ------------------------#

def hemmingSimilarity(state, goal):
    dim = int(sqrt(len(state)))
    assert dim ** 2 == len(state), "Puzzle board must be with square shape!"
    cost = 0
    for i, num in enumerate(state):
        if num == 0: continue   # skip the blank space
        j = i+1 if num <= goal else i
        cost += (num != j)
    return cost

def gaschingSimilarity(state, goal):
    N = len(state)
    curr_state = list(state)
    goal_idx = [v-1 if v <= goal else v for v in range(1, N)]
    goal_idx.insert(0, goal)
    i = 0
    while True:
        idx = [curr_state.index(i) for i in range(N)]
        if idx == goal_idx: return i
        if idx[0] < N-1:
            curr_state[idx[0]], curr_state[idx[idx[0]+1]] = curr_state[idx[idx[0]+1]], curr_state[idx[0]]
        else:
            for wrong_index in range(N):
                if idx[wrong_index] != goal_idx[wrong_index]: break
            curr_state[idx[0]], curr_state[idx[wrong_index]] = curr_state[idx[wrong_index]], curr_state[idx[0]]
        i += 1

def misplacedSimilarity(state, goal):
    numbers, rows, cols = state
    assert len(numbers) == rows*cols, "Error! Game state shape doesn't match!"
    cost = 0
    for i, num in enumerate(numbers):
        if num == 0: continue   # skip the blank space
        j = num-1 if num < goal else num
        # cost += sum(divmod(abs(i-j), dim))
        cost += (abs(i // cols - j // cols) + abs(i % cols - j % cols))
    return cost

def _disjointPatternsSimilarity():
    patterns = {}
    def _eval_similarity(state, goal):
        N = len(state)
        size = int(sqrt(N))
        index_tuples = [list(range(i * size, min((i+1)*size, N))) for i in range(N//size + bool(N%size))]
        relaxed_states = [tuple(v if id in idxs else -1 for id, v in enumerate(state))
                            for idxs in index_tuples]
        total_cost = 0
        for relaxed_state in relaxed_states:
            try:
                cost = patterns[relaxed_state]
            except KeyError:
                cost = relaxedSearch(relaxed_state, goal)
                patterns[relaxed_state] = cost
            total_cost += cost

        max_nodes = 2_000_000
        if len(patterns) >= max_nodes:
            print("too many patterns stored...")
            print(f"total cost of state {state} is {total_cost}")
            print("exiting...")
            exit(1)
        return total_cost
    return _eval_similarity

disjointPatternsSimilarity = _disjointPatternsSimilarity()


def relaxedSearch(relaxed_state, goal):
    relaxed_problem = RelaxedSlidingBlocks(relaxed_state, goal)
    problem = BlocksSearchProblem(relaxed_problem)   # make a new search problem
    path = astar(problem, misplacedSimilarity, track=True)  # find a path
    # path = ids(problem, track=True)
    return len(path)

#----------------------------------- Relaxed problems -----------------------------------#

class RelaxedSlidingBlocks(SlidingBlocksBoard):
    """ A relaxed sliding blocks board contains only a subset of the numbers of the entire
    board. For example if we have a 3x3 board with numbers (1, 0, 2, 3, 4, 5, 6, 7, 8),
    then  a relaxed board would be (1, 0, -1, 3, -1, 5, -1, 7, -1) having only odd numbers.
            -------------            -------------
            | 1 |   | 2 |            | 1 | 0 | * |
            -------------            -------------
            | 3 | 4 | 5 |            | 3 | * | 5 |
            -------------            -------------
            | 6 | 7 | 8 |            | * | 7 | * |
            -------------            -------------
    """
    def __init__(self, numbers, goal):
        """ Initialize a relaxed sliding blocks board.

        @param numbers (List[int]): A list of numbers representing the order of the
            numbers on the square grid. The number `0` represents the blank space.
        @param goal (int): The position of the blank space, after all the
            numbers are sorted in ascending order.
        """
        N = len(numbers) - 1
        dim = int(sqrt(N + 1))
        assert dim ** 2 == N + 1, "The puzzle must contain N^2 numbers!"
        self._dim = dim
        self._N = N
        self._cells = [list(numbers[i*dim : (i+1)*dim]) for i in range(dim)]
        self._goal = N if goal == -1 else goal
        self._goalState = list(range(1, N+1))
        self._goalState.insert(self._goal, 0)
        for idx, num in enumerate(self._goalState):
            if num not in numbers:
                self._goalState[idx] = -1
        self._goalState = tuple(self._goalState)

    @SlidingBlocksBoard.numbers.setter
    def numbers(self, input):
        self._cells = [list(input[i*self._dim : (i+1)*self._dim]) for i in range(self._dim)]

    def legalMoves(self):
        """ Moves consist of moving the number at position `i` one space up, down, left or
        right. These are encoded as (i, 'up'), (i, 'down'), (i, 'left') and (i, 'right')
        for i in range(N) respectively.
        """
        moves = []
        for i in range(self._N+1):
            if self.numbers[i] == -1:
                continue
            row, col = divmod(i, self._dim)
            if row != 0 and self._cells[row-1][col] in (-1, 0):
                moves.append((i, "up"))
            if row != self._dim-1 and self._cells[row+1][col] in (-1, 0):
                moves.append((i, "down"))
            if col != 0 and self._cells[row][col-1] in (-1, 0):
                moves.append((i, "left"))
            if col != self._dim-1 and self._cells[row][col+1] in (-1, 0):
                moves.append((i, "right"))
        return moves

    def result(self, move):
        """ Returns a new RelaxedSlidingBlocks. The move should be drawn legalMoves(). """
        idx, direction = move
        row, col = divmod(idx, self._dim)
        if direction == "up":
            newrow = row - 1
            newcol = col
        elif direction == "down":
            newrow = row + 1
            newcol = col
        elif direction == "left":
            newrow = row
            newcol = col - 1
        elif direction == "right":
            newrow = row
            newcol = col + 1
        else:
            raise "Illegal Move"
        # Create a new RelaxedSlidingBlocks and update it to reflect the move.
        newPuzzle = RelaxedSlidingBlocks(self.numbers, self._goal)
        newPuzzle._cells[row][col] = self._cells[newrow][newcol]
        newPuzzle._cells[newrow][newcol] = self._cells[row][col]
        return newPuzzle

#