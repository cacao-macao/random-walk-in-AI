import random

class SlidingBlocksActions:
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

class SlidingBlocksBoard:
    """ The game starts with a board consisting of blocks numbered 1 through N and one
    blank block represented by the number 0. The goal is to arrange the tiles according to
    their numbers. Moving is done by moving the blocks on top, bottom, left and right in
    place of the empty block.
    """

    def __init__(self, numbers, rows, cols, goal=-1):
        """ Initialize the sliding blocks board.

        @param numbers (List[int]): A list of numbers representing the order of the
            numbers on the square grid. The number `0` represents the blank space.
        @param rows (int): Number of rows on the puzzle board.
        @param cols (int): Number of columns on the puzzle board.
        @param goal (int): The position of the blank space, after all the
            numbers are sorted in ascending order.

        The list [1, 0, 2, 3, 4, 5, 6, 7, 8] represents the following 3x3 puzzle.
        And goal = (2,2) asks for the following solution:
            -------------            -------------
            | 1 |   | 2 |            | 1 | 2 | 3 |
            -------------            -------------
            | 3 | 4 | 5 |            | 4 | 5 | 6 |
            -------------            -------------
            | 6 | 7 | 8 |            | 7 | 8 |   |
            -------------            -------------
        """
        N = len(numbers)
        self._rows = rows
        self._cols = cols
        self._N = N
        self._cells = [list(numbers[i*cols : (i+1)*cols]) for i in range(rows)]
        self._goal = N-1 if goal == -1 else goal
        self._goalState = list(range(1, N))
        self._goalState.insert(self._goal, 0)
        self._goalState = tuple(self._goalState)
        self.blankLocation = numbers.index(0)

    @property
    def goal(self): return self._goal
    @property
    def numbers(self): return tuple(v for values in self._cells for v in values)
    @numbers.setter
    def numbers(self, input):
        assert len(input) == (self._N), "Error! Input must be the same shape."
        self._cells = [list(input[i*self._cols : (i+1)*self._cols]) for i in range(self._rows)]
        self.blankLocation = input.index(0)

    def isGoal(self):
        """ Check to see if the puzzle is in its goal state. """
        return self.numbers == self._goalState

    def legalMoves(self):
        """ Returns a list of legal moves from the current state.
        Moves consist of moving the blank space up, down, left or right. These are encoded
        as 'up', 'down', 'left' and 'right' respectively.
        """
        moves = []
        row, col = divmod(self.blankLocation, self._cols)
        if row != 0:
            moves.append(SlidingBlocksActions.UP)
        if row != self._rows-1:
            moves.append(SlidingBlocksActions.DOWN)
        if col != 0:
            moves.append(SlidingBlocksActions.LEFT)
        if col != self._cols-1:
            moves.append(SlidingBlocksActions.RIGHT)
        return moves

    def result(self, move):
        """ Returns a new SlidingBlocksBoard with the current state and blankLocation
        updated based on the provided move.
        The move should be a string drawn from a list returned by legalMoves. Illegal
        moves will raise an exception, which may be an array bounds exception.
        NOTE: This function *does not* change the current object. Instead, it returns a
        new object.
        """
        row, col = divmod(self.blankLocation, self._cols)
        if move == SlidingBlocksActions.UP:
            newrow = row - 1
            newcol = col
        elif move == SlidingBlocksActions.DOWN:
            newrow = row + 1
            newcol = col
        elif move == SlidingBlocksActions.LEFT:
            newrow = row
            newcol = col - 1
        elif move == SlidingBlocksActions.RIGHT:
            newrow = row
            newcol = col + 1
        else:
            raise "Illegal Move"
        # Create a new SlidingBlocksBoard and update it to reflect the move.
        newPuzzle = SlidingBlocksBoard(self.numbers, self._rows, self._cols, self._goal)
        newPuzzle._cells[row][col] = self._cells[newrow][newcol]
        newPuzzle._cells[newrow][newcol] = self._cells[row][col]
        newPuzzle.blankLocation = newrow * self._cols + newcol
        return newPuzzle

    def __eq__(self, other):
        """ Overloads '==' such that two SlidingBlocksBoard with the same configuration
        are equal.

          >>> SlidingBlocksBoard([0, 1, 2, 3, 4, 5, 6, 7, 8]) == \
              SlidingBlocksBoard([1, 0, 2, 3, 4, 5, 6, 7, 8]).result('left')
          True
        """
        return self.numbers == other.numbers

    def __hash__(self):
        return hash(str(self.numbers))

    def __getAsciiString(self):
        """ Returns a display string for the puzzle. """
        num_digits = max(len(str(v)) for v in self.numbers)
        lines = []
        horizontalLine = ('-' * (self._cols*(num_digits+3)+1))
        lines.append(horizontalLine)
        for row in self._cells:
            rowLine = "|"
            for d in row:
                if d == 0:
                    rowLine = rowLine + " " + " "*num_digits + " |"
                else:
                    rowLine = rowLine + f" {d:>{num_digits}d} |"
            lines.append(rowLine)
            lines.append(horizontalLine)
        return "\n".join(lines)

    def __str__(self):
        return self.__getAsciiString()


def create_random_puzzle(seed=0, rows=3, cols=3, goal=-1):
    """ Creates a random sliding blocks puzzle by applying a series of random moves to a
    solved puzzle.

    @param seed (int): Seed for random number generation.
    @param rows (int): Number of rows on the puzzle board.
    @param cols (int): Number of columns on the puzzle board.
    @param moves (int): Number of random moves to apply.
    """
    random.seed(seed)
    N = rows*cols
    goalState = list(range(1, N))
    goalState.insert(goal, 0)
    puzzle = SlidingBlocksBoard(goalState, rows, cols, goal)
    moves = 100*N
    for _ in range(moves): # Execute random legal moves
        puzzle = puzzle.result(random.sample(puzzle.legalMoves(), 1)[0])
    return puzzle

def create_puzzle(numbers, rows, cols, goal=-1):
    return SlidingBlocksBoard(numbers, rows, cols, goal)

#