import argparse
import sys
sys.path.append("../")


from games.agents import AdversarialAgent
from games.minimax import minimax, alpha_beta, expectimax
from games.pacman_game import PacmanGameState, scoreEvaluationFunction, betterEvaluationFunction
from games.tictactoe_game import TicTacToeGameState, evaluate


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--numgames", dest="N", type=int,
                    help="number of games to play for Pacman or size of the puzzle", default=1)
parser.add_argument("-l", "--layout", dest="layout", type=str,
                    help="map layout to be loaded",  default="mediumClassic")
parser.add_argument("-g", "--ghost", dest="ghost", type=str,
                    help="the ghost agent type to use", default="RandomGhost")
parser.add_argument("-k", "--numghosts", dest="numGhosts", type=int,
                    help="maximum number of ghosts to use", default=4)
parser.add_argument("-z", "--zoom", dest="zoom", type=float,
                    help="zoom the size of the graphics window", default=1.0)
parser.add_argument("--frameTime", dest="frameTime", type=float,
                    help="time delay between frames; <0 means keyboard", default=0.1)
parser.add_argument("-r", "--random", dest="fixRandomSeed", action="store_true",
                    help="fix the random seed to always play the same game", default=False)
parser.add_argument("-t", "--textGraphics", dest="textGraphics", action="store_true",
                    help="display output as text only", default=False)
parser.add_argument("-q", "--quietGraphics", dest="quietGraphics", action="store_true",
                    help="generate minimal output and no graphics", default=False)

parser.add_argument("-s", "--seed", dest="seed", type=int, help="random seed value", default=0)
parser.add_argument("-p", "--problem", dest="problem", type=str,
                    help="game type", default="pacman")
parser.add_argument("-m", "--method", dest="method", type=str,
                    help="game search method used", default="")
parser.add_argument("--heuristic", dest="heuristic", type=str,
                    help="utility function used to evaluate game states", default="None")
parser.add_argument("--agent_id", dest="agent_id", type=int,
                    help="Chose whether the agent goes first (id=0) or second (id=1)", default=0)
parser.add_argument("--depth", dest="depth", type=int,
                    help="depth of the game search tree", default=float("inf"))
args = parser.parse_args()


# parser.add_option('-r', '--recordActions', action='store_true', dest='record',
#                   help='Writes game histories to a file (named by the time they were played)', default=False)
# parser.add_option('--replay', dest='gameToReplay',
#                   help='A recorded game file (pickle) to replay', default=None)
# parser.add_option('-x', '--numTraining', dest='numTraining', type='int',
#                   help=default('How many episodes are training (suppresses output)'), default=0)
# parser.add_option('-c', '--catchExceptions', action='store_true', dest='catchExceptions',
#                   help='Turns on exception handling and timeouts during games', default=False)
# parser.add_option('--timeout', dest='timeout', type='int',
#                   help=default('Maximum length of time an agent can spend computing in a single game'), default=30)

# Init some more args.
args.record = False
args.gameToReplay = None
args.numTraining = 0
args.catchExceptions = False
args.timeout = 30


# Map command line arguments.
_GAMES = {
    "pacman"        : PacmanGameState,
    "tictactoe"     : TicTacToeGameState,
}
_GAMESEARCH_METHODS = {
    "minimax"       : minimax,
    "alphabeta"     : alpha_beta,
    "expectimax"    : expectimax,
}
_UTILITY = {
    "None"          : None,
    "score"         : scoreEvaluationFunction,
    "better"        : betterEvaluationFunction,
    "evaluate"      : evaluate,
}


# Create a search agent.
if args.method != "":
    args.agent = AdversarialAgent(_GAMES[args.problem],
        _GAMESEARCH_METHODS[args.method], _UTILITY[args.heuristic], depth=args.depth)


# Run the game.
if args.problem == "pacman":
    from envs.pacman.pacman import run
if args.problem == "tictactoe":
    from games.tictactoe_game import run

run(args)

#