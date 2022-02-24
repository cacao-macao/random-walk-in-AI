import time
import sys
sys.path.append("..")


from satisfaction.agents import SatisfactionAgent
from satisfaction.base import BaseSatisfactionProblem
from satisfaction.local_methods import hill_climb, min_conflicts, simulated_annealing
from satisfaction.nqueens_satisfaction import NQueensSatisfaction
from satisfaction.satisfaction_methods import arc_consitency, backtrack, forward_checking, fwdMRV, arcMRV


# Map of Australia
variables = ["WA", "SA", "NT", "NSW", "Q", "V", "T"]
neighbors = {"WA":["NT","SA"],
             "NT":["WA","SA","Q"],
             "SA":["WA", "NT", "Q", "NSW", "V"],
             "Q":["NT","SA","NSW"],
             "NSW":["Q","SA","V"],
             "V":["NSW","SA"],
             "T":[]
            }
colors = "RGB"
domains = {var:set(colors) for var in variables}
constraints = lambda var1, val1, var2, val2: \
    (var1 == var2) or (var2 not in neighbors[var1]) or (val1 != val2)
agent = SatisfactionAgent(BaseSatisfactionProblem, backtrack)
agent.registerCSP(variables, domains, neighbors, constraints)
print(f"Solution to the Australia map coloring problem:\n{agent.assignments}\n")


# Map of USA
variables = ['AL', 'AK', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL',
             'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KA', 'KY', 'LA', 'MA',
             'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE',
             'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI',
             'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV',
             'WY']
neighbors = {
    "HI": [],
    "AK": [],
    "WA": ['OR', 'ID'],
    "OR": ['WA', 'ID', 'NV', 'CA'],
    "ID": ['WA', 'OR', 'NV', 'MT', 'WY', 'UT'],
    "NV": ['OR', 'CA', 'ID', 'UT', 'AZ'],
    "CA": ['OR', 'NV', 'AZ'],
    "AZ": ['CA', 'NV', 'UT', 'NM'],
    "UT": ['NV', 'ID', 'WY', 'CO', 'AZ'],
    "MT": ['ID', 'ND', 'SD', 'WY'],
    "WY": ['ID', 'UT', 'MT', 'SD', 'NE', 'CO'],
    "CO": ['UT', 'WY', 'NE', 'KA', 'OK', 'NM'],
    "ND": ['MT', 'MN', 'SD'],
    "SD": ['MT', 'WY', 'ND', 'MN', 'IA', 'NE'],
    "NE": ['WY', 'CO', 'SD', 'IA', 'MO', 'KA'],
    "KA": ['CO', 'NE', 'MO', 'OK'],
    "OK": ['CO', 'NM', 'KA', 'MO', 'AR', 'TX'],
    "NM": ['CO', 'OK', 'TX', 'AZ'],
    "TX": ['NM', 'OK', 'AR', 'LA'],
    "MN": ['ND', 'SD', 'WI', 'IA'],
    "IA": ['SD', 'NE', 'MN', 'WI', 'IL', 'MO'],
    "MO": ['NE', 'KA', 'OK', 'IA', 'IL', 'KY', 'TN', 'AR'],
    "AR": ['OK', 'TX', 'MO', 'MS', 'TN', 'LA'],
    "LA": ['TX', 'AR', 'MS'],
    "WI": ['MN', 'IA', 'MI', 'IL'],
    "IL": ['IA', 'MO', 'WI', 'IN', 'KY'],
    "KY": ['MO', 'IL', 'IN', 'OH', 'WV', 'VA', 'TN'],
    "TN": ['MO', 'AR', 'MS', 'AL', 'KY', 'VA', 'NC', 'GA'],
    "MS": ['AR', 'LA', 'TN', 'AL'],
    "MI": ['WI', 'OH', 'IN'],
    "IN": ['IL', 'OH', 'KY', 'MI'],
    "OH": ['IN', 'MI', 'PA', 'WV', 'KY'],
    "AL": ['MS', 'TN', 'GA', 'FL'],
    "GA": ['AL', 'TN', 'NC', 'SC', 'FL'],
    "FL": ['AL', 'GA'],
    "PA": ['OH', 'NY', 'NJ', 'DE', 'MD', 'WV'],
    "WV": ['OH', 'KY', 'PA', 'MD', 'VA'],
    "VA": ['KY', 'TN', 'WV', 'MD', 'DC', 'NC'],
    "NC": ['TN', 'GA', 'VA', 'SC'],
    "SC": ['GA', 'NC'],
    "NY": ['PA', 'VT', 'MA', 'CT', 'NJ'],
    "NJ": ['PA', 'NY', 'DE'],
    "DE": ['PA', 'NJ', 'MD'],
    "MD": ['PA', 'WV', 'VA', 'DE', 'DC'],
    "DC": ['VA', 'MD'],
    "VT": ['NY', 'NH', 'MA'],
    "MA": ['NY', 'VT', 'NH', 'RI', 'CT'],
    "CT": ['NY', 'MA', 'RI'],
    "NH": ['VT', 'MA', 'ME'],
    "RI": ['MA', 'CT'],
    "ME": ['NH'],
}
colors = "RGB"
domains = {var:set(colors) for var in variables}
constraints = lambda var1, val1, var2, val2: \
    (var1 == var2) or (var2 not in neighbors[var1]) or (val1 != val2)
agent = SatisfactionAgent(BaseSatisfactionProblem, arc_consitency)
agent.registerCSP(variables, domains, neighbors, constraints)
print(f"Solution to the USA map coloring problem with 3 colors:\n{agent.assignments}\n")
colors = "RGBY"
domains = {var:set(colors) for var in variables}
constraints = lambda var1, val1, var2, val2: \
    (var1 == var2) or (var2 not in neighbors[var1]) or (val1 != val2)
agent = SatisfactionAgent(BaseSatisfactionProblem, arc_consitency)
agent.registerCSP(variables, domains, neighbors, constraints)
print(f"Solution to the USA map coloring problem with 4 colors:\n{agent.assignments}\n")

# 8-Queens - backtracking
N = 8
variables = list(range(N))
domains = {var:set(range(N)) for var in variables}
neighbor = list(range(N))
neighbors = {var: neighbor for var in variables}
constraints = lambda var1, val1, var2, val2: \
            var1==var2 or (val1!=val2 and var1+val1!=var2+val2 and var1-val1!=var2-val2)
agent = SatisfactionAgent(NQueensSatisfaction, backtrack)
agent.registerCSP(variables, domains, neighbors, constraints)
print(f"Solution to the N-Queens problem for N={N}:\n{agent.assignments}\n")

# N-Queens - compare forward checking and arc consistency
N = 50
# variables = list(range(N))
# domains = {var:set(range(N)) for var in variables}
# neighbor = list(range(N))
# neighbors = {var: neighbor for var in variables}
# constraints = lambda var1, val1, var2, val2: \
#             var1==var2 or (val1!=val2 and var1+val1!=var2+val2 and var1-val1!=var2-val2)
# agent = SatisfactionAgent(BaseSatisfactionProblem, fwdMRV)
# print(f"Solving {N}-queens with forward checking with MRV")
# agent.registerCSP(variables, domains, neighbors, constraints)

# variables = list(range(N))
# domains = {var:set(range(N)) for var in variables}
# neighbor = list(range(N))
# neighbors = {var: neighbor for var in variables}
# constraints = lambda var1, val1, var2, val2: \
#             var1==var2 or (val1!=val2 and var1+val1!=var2+val2 and var1-val1!=var2-val2)
# agent = SatisfactionAgent(BaseSatisfactionProblem, forward_checking)
# print(f"\nSolving {N}-queens with forward checking with MRV and LCV")
# agent.registerCSP(variables, domains, neighbors, constraints)

variables = list(range(N))
domains = {var:set(range(N)) for var in variables}
neighbor = list(range(N))
neighbors = {var: neighbor for var in variables}
constraints = lambda var1, val1, var2, val2: \
            var1==var2 or (val1!=val2 and var1+val1!=var2+val2 and var1-val1!=var2-val2)
agent = SatisfactionAgent(NQueensSatisfaction, arcMRV)
print(f"\nSolving {N}-queens with arc consistency with MRV")
agent.registerCSP(variables, domains, neighbors, constraints)

variables = list(range(N))
domains = {var:set(range(N)) for var in variables}
neighbor = list(range(N))
neighbors = {var: neighbor for var in variables}
constraints = lambda var1, val1, var2, val2: \
            var1==var2 or (val1!=val2 and var1+val1!=var2+val2 and var1-val1!=var2-val2)
agent = SatisfactionAgent(NQueensSatisfaction, arc_consitency)
print(f"\nSolving {N}-queens with arc consistency with MRV and LCV")
agent.registerCSP(variables, domains, neighbors, constraints)
print()


# N-queens with hill-climb.
N = 50
variables = list(range(N))
domain = set((range(N)))
domains = {var: domain for var in variables}
neighbor = list(range(N))
neighbors = {var: neighbor for var in variables}
constraints = lambda var1, val1, var2, val2: \
            var1==var2 or (val1!=val2 and var1+val1!=var2+val2 and var1-val1!=var2-val2)
agent = SatisfactionAgent(NQueensSatisfaction, hill_climb)
tic = time.time()
agent.registerCSP(variables, domains, neighbors, constraints)
toc = time.time()
print(f"Solution to the N-Queens problem with hill-climbing for N={N} took {toc-tic:.3f} seconds\n")


# N-queens with simulated-annealing.
N = 50
variables = list(range(N))
domain = set((range(N)))
domains = {var: domain for var in variables}
neighbor = list(range(N))
neighbors = {var: neighbor for var in variables}
constraints = lambda var1, val1, var2, val2: \
            var1==var2 or (val1!=val2 and var1+val1!=var2+val2 and var1-val1!=var2-val2)
agent = SatisfactionAgent(NQueensSatisfaction, simulated_annealing)
tic = time.time()
agent.registerCSP(variables, domains, neighbors, constraints)
toc = time.time()
print(f"Solution to the N-Queens problem with simulated-annealing for N={N} took {toc-tic:.3f} seconds\n")


# N-Queens with min-conflicts
N = 10000
variables = list(range(N))
domain = set((range(N)))
domains = {var: domain for var in variables}
neighbor = list(range(N))
neighbors = {var: neighbor for var in variables}
constraints = lambda var1, val1, var2, val2: \
            var1==var2 or (val1!=val2 and var1+val1!=var2+val2 and var1-val1!=var2-val2)
agent = SatisfactionAgent(NQueensSatisfaction, min_conflicts)
tic = time.time()
agent.registerCSP(variables, domains, neighbors, constraints)
toc = time.time()
print(f"Solution to the N-Queens problem with min-conflicts for N={N} took {toc-tic:.3f} seconds\n")

#