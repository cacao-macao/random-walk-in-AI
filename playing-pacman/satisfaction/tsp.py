import random

from satisfaction.base import BaseSatisfactionProblem


class TravellingSalesmanProblem(BaseSatisfactionProblem):
    def __init__(self, N):
        self._variables = list(range(N))
        domain = list(range(N))
        self._domains = {var:domain for var in self._variables}
        neighbors = list(range(N))
        self._neighbors = {var:neighbors for var in self._variables}
        self.constraints = lambda *args: True
        self._distances = {}
        for i in self._variables:
            for j in self._variables:
                if i == j:
                    self._distances[(i, j)] = 0
                else:
                    self._distances[(i, j)] = random.random() * 1000
        self.reset()


def pathScore(assignment, distances):
    path = [0] * len(assignment)
    for city, number in assignment.items():
        path[number] = city
    cost = 0
    for i in range(len(path)-1):
        cost += distances[(path[i], path[i+1])]
    return cost