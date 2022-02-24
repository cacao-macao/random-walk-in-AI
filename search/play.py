import argparse
import sys
sys.path.append("../")


from search.agents import SearchAgent
from search.blocks_search_problems import BlocksSearchProblem
from search.blocks_search_problems import disjointPatternsSimilarity, gaschingSimilarity, hemmingSimilarity, misplacedSimilarity
from search.pacman_search_problems import CornersProblem, FoodSearchProblem, PositionSearchProblem
from search.pacman_search_problems import manhattanDistance, euclideanDistance
from search.search_methods import astar, bfs, dfs, ids, rbfs, ucs
from search.smaStar import smaStar



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
                    help="search problem type", default="pacmanPositionSearch")
parser.add_argument("-m", "--method", dest="method", type=str,
                    help="search method used", default="")
parser.add_argument("--heuristic", dest="heuristic", type=str,
                    help="heuristic used for the search method", default="None")
parser.add_argument("--rows", dest="rows", type=int, help="number of rows", default=3)
parser.add_argument("--cols", dest="cols", type=int, help="number of cols", default=3)
parser.add_argument("--input", dest="input", action='store_true', default=False)
parser.add_argument("--verbose", dest="verbose", action='store_true', default=False)

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
_SEARCH_PROBLEMS = {
    "blocks"                : BlocksSearchProblem,
    "pacmanPositionSearch"  : PositionSearchProblem,
    "pacmanCornerSearch"    : CornersProblem,
    "pacmanFoodSearch"      : FoodSearchProblem,
}
_SEARCH_METHODS = {
    "bfs"   : bfs,
    "dfs"   : dfs,
    "ucs"   : ucs,
    "astar" : astar,
    "ids"   : ids,
    "rbfs"  : rbfs,
    "sma"   : smaStar,
}
_SEARCH_HEURISTICS = {
    "None"      : None,
    "hemming"   : hemmingSimilarity,            # heuristic for sliding blocks
    "gasching"  : gaschingSimilarity,           # heuristic for sliding blocks
    "misplaced" : misplacedSimilarity,          # heuristic for sliding blocks
    "disjoint"  : disjointPatternsSimilarity,   # heuristic for sliding blocks
    "manhattan" : manhattanDistance,    # heuristic for pacman
    "euclidean" : euclideanDistance,    # heuristic for pacman
}


# Create a search agent.
if args.method != "":
    args.agent = SearchAgent(_SEARCH_PROBLEMS[args.problem],
        _SEARCH_METHODS[args.method], _SEARCH_HEURISTICS[args.heuristic], args.verbose)


# Run the game.
if "blocks" in args.problem:
    from envs.blocks.gui import run
if "pacman" in args.problem:
    from envs.pacman.pacman import run

run(args)

#