import argparse
import sys
sys.path.append("..")


from satisfaction.agents import SatisfactionAgent
from satisfaction.base import BaseSatisfactionProblem
from satisfaction.local_methods import hill_climb, min_conflicts, simulated_annealing
from satisfaction.nqueens_satisfaction import NQueensSatisfaction, printNQueens
from satisfaction.satisfaction_methods import arc_consitency, backtrack, forward_checking


# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("-n", "--numgames", dest="N", type=int,
                    help="number of games to play for Pacman or size of the puzzle", default=1)
parser.add_argument("-s", "--seed", dest="seed", type=int, help="random seed value", default=0)
parser.add_argument("-p", "--problem", dest="problem", type=str,
                    help="search problem type", default="nqueens")
parser.add_argument("-m", "--method", dest="method", type=str,
                    help="search method used", default="")
parser.add_argument("--verbose", dest="verbose", action='store_true', default=False)
args = parser.parse_args()


# Map command line arguments.
_SEARCH_PROBLEMS = {
    "mapcolor"      : BaseSatisfactionProblem,
    "nqueens"       : NQueensSatisfaction,
}
_SEARCH_METHODS = {
    "backtrack"     : backtrack,
    "forward"       : forward_checking,
    "arc"           : arc_consitency,
    "hillclimb"     : hill_climb,
    "minconflicts"  : min_conflicts,
    "annealing"     : simulated_annealing,
}


# Create a satisfaction agent.
agent = SatisfactionAgent(_SEARCH_PROBLEMS[args.problem], _SEARCH_METHODS[args.method])


# Create a satisfaction problem.
if args.problem == "nqueens":
    variables = list(range(args.N))
    domain = set((range(args.N)))
    domains = {var: domain for var in variables}
    neighbor = list(range(args.N))
    neighbors = {var: neighbor for var in variables}
    constraints = lambda var1, val1, var2, val2: \
                var1==var2 or (val1!=val2 and var1+val1!=var2+val2 and var1-val1!=var2-val2)
elif args.problem == "mapcolor":
    variables = ["WA", "SA", "NT", "NSW", "Q", "V", "T"]
    neighbors = {"WA" : ["NT","SA"],
                 "NT" : ["WA","SA","Q"],
                 "SA" : ["WA", "NT", "Q", "NSW", "V"],
                 "Q"  : ["NT","SA","NSW"],
                 "NSW": ["Q","SA","V"],
                 "V"  : ["NSW","SA"],
                 "T"  : []
                }
    colors = "RGB"
    domains = {var:set(colors) for var in variables}
    constraints = lambda var1, val1, var2, val2: \
        (var1 == var2) or (var2 not in neighbors[var1]) or (val1 != val2)


# Run the agent.
agent.registerCSP(variables, domains, neighbors, constraints)

if args.verbose:
    print(agent.assignments)
    if args.problem == "nqueens":
        printNQueens(agent.assignments)

#