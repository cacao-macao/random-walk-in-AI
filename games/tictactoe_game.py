from games.base import BaseGameState


class TicTacToeGameState(BaseGameState):

    def __init__(self, config=None, agent_ID=0, N=3):
        """ Initialize the tic-tac-toe board.

        The state of the game is represented by a string of 'X's, 'O's and '-'s.
        -------------
        | x | - | o |
        -------------
        | x | o | - |    =  (x-oxo----)
        -------------
        | - | - | - |
        -------------

        @param N (int): The size of the board will be N x N.
        """
        if config is None:
            self._N = N
            self._state = "-" * N**2
            self._empty_slots = N**2
            self._to_move = 0
            self._agent_ID = agent_ID
        else:
            self._N = config["N"]
            self._state = config["state"]
            self._empty_slots = config["empty_slots"]
            self._to_move = config["to_move"]
            self._agent_ID = config["agent_ID"]

    def actions(self):
        """ Returns a list of legal actions for the player which is to move next. """
        return [i for i, x in enumerate(self._state) if x == "-"]

    def to_move(self):
        """ Return the id of the player whose turn it is to play next. """
        return self._to_move

    def is_terminal(self):
        """ Returns true if the game state is a terminal state."""
        return self._isTerminal() or self._empty_slots == 0

    def _isTerminal(self):
        N = self._N
        state = self._state
        if all(state[i*N + i] != '-' and state[i*N + i] == state[0] for i in range(N)): # check main diagonal
            return True
        if all(state[i*N+(N-1-i)] != '-' and state[i*N+(N-1-i)] == state[N-1] for i in range(N)): # check secondary diagonal
            return True
        for i in range(N):
            if all(x != '-' and x == state[N*i] for x in state[N*i:N*i+N]): # check rows
                return True
            if all(x != '-' and x == state[i] for x in state[i::N]): # check columns
                return True
        return False

    def agent_id(self):
        """ Returns the ID of the player controlled by the agent. """
        return self._agent_ID

    def score(self):
        """ Return the score of the game state. """
        if not self._isTerminal() and self._empty_slots == 0:
            return 0
        return -1 if (self.to_move() == self._agent_ID) else (1 + self._empty_slots)

    def result(self, action):
        """ Returns the successor game state after a player takes an action.

        @param action (Tuple(int, int)): A tuple (row, col) giving the row and column
        numbers where the player wants play.
        @return result (gameState): The resulting game state.
        """
        if self._state[action] != "-":
            raise ValueError("Illegal action")
        N = self._N
        row, col = divmod(action, N)
        player = self._to_move
        mark = 'X' if player == 0 else 'O'
        new_state = TicTacToeGameState(agent_ID=self._agent_ID, N=self._N)
        new_state._state = self._state[:row*N+col] + mark + self._state[row*N+col+1:]
        new_state._empty_slots = self._empty_slots - 1
        new_state._to_move = (self._to_move + 1) % 2
        return new_state

    def __hash__(self):
        return hash(self._state)

#------------------------------------ Utility funcs -------------------------------------#
def evaluate(gameState):
    return gameState.score()

#------------------------------------- Helper funcs -------------------------------------#
def run(args):
    try:
        agent = args.agent
        if args.agent_id == 0:
            player1 = lambda game:agent_move(agent, game)
            player2 = player_move
        else:
            player1 = player_move
            player2 = lambda game:agent_move(agent, game)
    except AttributeError:
        player1 = player_move
        player2 = player_move

    game = TicTacToeGameState(agent_ID=args.agent_id, N=args.N)
    while True:
        game = player1(game)
        if game.is_terminal(): return
        game = player2(game)
        if game.is_terminal(): return

def player_move(game):
    while True:
        action = tuple(map(int, input("Pick row and column where to play\n").split(",")))
        row, col = action
        action = row * game._N + col
        try:
            game = game.result(action)
            break
        except ValueError:
            print("Illegal action! Please, pick another action.")
    printState(game)
    if game._isTerminal():
        print("You win!")
        return game
    elif game._empty_slots == 0:
        print("The game ends in a draw!")
        return game
    return game

def agent_move(agent, game):
    config = {"N":game._N, "state":game._state, "empty_slots":game._empty_slots,
            "to_move":game._to_move, "agent_ID":game._agent_ID}
    act = agent.getAction(config)
    game = game.result(act)
    print("The agent picked an action...")
    printState(game)
    if game._isTerminal():
        print("You lose!")
        return game
    elif game._empty_slots == 0:
        print("The game ends in a draw!")
        return game
    return game

def printState(gameState):
    """ Returns a display string for the game. """
    N = gameState._N
    lines = []
    horizontalLine = '-' * (4*N+1)
    current_line = []
    for i, mark in enumerate(gameState._state):
        current_line.append("| " + mark + " ")
        if (i+1) % N == 0:
            lines.append(horizontalLine)
            lines.append(''.join(current_line) + "|")
            current_line = []
    lines.append(horizontalLine)
    print("\n".join(lines))

#