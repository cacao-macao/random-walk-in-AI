import argparse
import random
import sys
sys.path.append("../")

from mdp.agents import ValueIterationAgent
from mdp.gridworld_mdp import GridWorldMDP
from mdp.value_iteration import (asynchronous_value_iteration, full_value_iteration,
                                 priority_sweep_value_iteration)

# Parse command line arguments.
parser = argparse.ArgumentParser()
# Grid world arguments.
parser.add_argument("--discount", dest="discount", type=float, default=0.9,
                    help="Discount on future (default %default)")
parser.add_argument("--livingReward", dest="livingReward", type=float, default=0.0,
                    help="Reward for living for a time step (default %default)")
parser.add_argument("--noise", dest="noise", type=float, default=0.2,
                    help="How often action results in unintended direction (default %default)")
parser.add_argument("--grid", dest="grid", type=str, default="BookGrid",
                    help="Grid to use (case sensitive; options are BookGrid, BridgeGrid, CliffGrid, MazeGrid, default %default)")
parser.add_argument("--windowSize", dest="gridSize", type=int, default=150,
                    help="Request a window width of X pixels *per grid cell* (default %default)")
parser.add_argument("--speed", dest="speed", type=float, default=1.0,
                    help="Speed of animation, S > 1.0 is faster, 0.0 < S < 1.0 is slower (default %default)")
parser.add_argument("--text", dest="textDisplay", action="store_true", default=False,
                    help='Use text-only ASCII display')
parser.add_argument("--pause", dest="pause", action='store_true', default=False,
                    help="Pause GUI after each time step when running the MDP")
parser.add_argument("--quiet", dest="quiet", action="store_true", default=False,
                    help="Skip display of any learning episodes")
parser.add_argument("--valueSteps", dest="valueSteps", action="store_true", default=False,
                    help="Display each step of value iteration")

parser.add_argument('--manual',action='store_true',
                         dest='manual',default=False,
                         help='Manually control agent')

parser.add_argument("-s", "--seed", dest="seed", type=int, help="random seed value", default=0)
parser.add_argument("-p", "--problem", dest="problem", type=str,
                    help="MDP environment", default="gridworld")
parser.add_argument("-m", "--method", dest="method", type=str,
                    help="value iteration method used", default="")
parser.add_argument("--epsi", dest="epsi", type=float, default=0.3,
                    help="Chance of taking a random action in q-learning (default %default)")
parser.add_argument("--iters", dest="iters", type=int, default=10,
                    help="Number of iterations of value iteration (default %default)")
parser.add_argument("--episodes", dest="episodes", type=int, default=1,
                    help="Number of episodes of the MDP to run (default %default)")
parser.add_argument("--train", dest="train", action="store_true", default=False,
                    help="Train the agent")
args = parser.parse_args()


random.seed(args.seed)


# Map command line arguments.
_MDPs = {
    "gridworld" :   GridWorldMDP,
}
_MDP_AGENTS = {
    "value_iteration"   :   ValueIterationAgent,
}
_MDP_METHODS = {
    "full"          :   full_value_iteration,
    "async"         :   asynchronous_value_iteration,
    "prioritized"   :   priority_sweep_value_iteration,
}


# Create an MDP agent.
if args.method != "":
    args.agent = ValueIterationAgent(_MDPs[args.problem], _MDP_METHODS[args.method])
else:
    args.agent = None


# Run the game.
if "gridworld" in args.problem:
    from envs.gridworld.gridworld import run
if "pacman" in args.problem:
    from envs.pacman.pacman import run

run(args)

#