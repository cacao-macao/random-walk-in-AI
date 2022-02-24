import random
import time


class GeneticSearch:
    def __init__(self, problem, population_size, mutation_rate):
        """ Initialize a genetic search instance.

        @param problem (problem): A factor graph problem instance.
        @param population_size (int): Size of the population.
        @param mutation_rate (float): The probability for a random mutation of a single gene.
        """
        self._problem = problem
        self._size = population_size
        self._p = mutation_rate
        self._variables = problem.getVariables()
        self._population = [None] * population_size

    def start(self, num_iters, fitness, threshold, seed=0):
        """ Start the iterative genetic search procedure.

        # Select the next state by choosing a variable and a new value to be assigned to
        # that variable. If the new assignment improves the csp, then the variable is
        # re-assigned, otherwise the algorithm accepts the re-assignment with some
        # probability less than 1. The probability decreases exponentially with the `badness`
        # of the re-assignment and the temperature.
    
        @param num_iters (int): Number of iterations to run the search for.
        @param fitness (func): Function for evaluating individuals from the population.
        @param seed (float): Initial value of the random seed parameter.
        # @return assignment (dict{var:val}): A complete assignment to the csp that
        #     satisfies all constraints. If the local search method doesn't find a solution,
        #     then return an empty assignment ({}).
        """
        # Initialize the population with a set of complete assignments.
        random.seed(seed)
        for i in range(self._size):
            self._problem.random_assignments()
            assignments = self._problem.getAssignments()
            self._population[i] = [assignments[var] for var in self._variables]

        # Run the iterative local search procedure.
        for _ in range(num_iters):
            # Check if a solution is found.
            for p in self._population:
                if fitness(p) == threshold:
                    for i in range(len(self._variables)):
                        self._problem.assign(self._variables[i], p[i])
                    print("found a solution to the problem... returning")
                    return

            # Select the individuals who will become the parents of the next generation.
            parent_pairs = self._select(fitness)

            # Produce offspring that populate the next generation using recombination.
            offspring = []
            for pair in parent_pairs:
                offspring.extend(self._crossover(pair))
            
            # Perform random mutations.
            self._population = [self._mutate(child) for child in offspring]

        # Use the best individual from the population.
        best = max(self._population, key=fitness)
        self._problem.reset()
        for i in range(len(self._variables)):
            self._problem.assign(self._variables[i], best[i])

    def _select(self, fitness):
        population = self._population
        dist = list(map(fitness, population))
        parent_pairs = []
        for _ in range(self._size // 2):
            parents = random.choices(population, dist, k=2)
            parent_pairs.append(parents)
        return parent_pairs

    def _crossover(self, parents):
        v, w = parents
        c = random.randint(0, len(v)-1)
        return v[:c]+w[c:], w[:c]+v[c:]

    def _mutate(self, individual):
        N = len(individual)
        for i in range(N):
            if flipCoin(self._p):
                domain = list(self._problem.getDomain(individual[i]))
                individual[i] = random.choice(domain)
        return individual


def flipCoin(p):
    r = random.random()
    return r < p

#


if __name__ == "__main__":

    import sys
    sys.path.append("..")

    from satisfaction.nqueens_satisfaction import NQueensSatisfaction, printBoard, nonatacking_pairs

    def nonatacking_pairs_gen(individual):
        count = 0
        N = len(individual)
        for i in range(N-1):
            for j in range(i+1, N):
                v = individual[i]
                w = individual[j]
                if nqueens.constraints(i, v, j, w):
                    count += 1
        return count
    
    def attacking_pairs_gen(individual):
        count = 0
        not_count = 0
        N = len(individual)
        for i in range(N-1):
            for j in range(i+1, N):
                v = individual[i]
                w = individual[j]
                if not nqueens.constraints(i, v, j, w):
                    count += 1
                if nqueens.constraints(i, v, j, w):
                    not_count += 1
        return  1 / (count + 1)


    fitness = attacking_pairs_gen

    N = 8
    variables = list(range(N))
    domains = {var:set(range(N)) for var in variables}
    neighbor = list(range(N))
    neighbors = {var: neighbor for var in variables}
    constraints = lambda var1, val1, var2, val2: \
                var1==var2 or (val1!=val2 and var1+val1!=var2+val2 and var1-val1!=var2-val2)
    nqueens = NQueensSatisfaction(variables, domains, neighbors, constraints)
    g = GeneticSearch(nqueens, 100, 0.01)
    num_iters = 1000
    tic = time.time()
    g.start(num_iters, fitness)
    toc = time.time()
    print(f"{num_iters} iterations take {toc-tic:.3f} seconds")
    print("best individual after evolution:", fitness(g._problem.getAssignments()))

