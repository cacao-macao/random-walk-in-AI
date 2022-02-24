import math
import random


class LocalSearch:
    """ General implementation of a local search algorithm.
    Searches for a solution to the given constraint satisfaction problem by initializing
    a start state and searching to neighboring states. The search is not systematic and
    might never explore a portion of the search space where a solution actually resides.
    However, it can often find reasonable solutions in large or infinite state spaces.   
    """

    def __init__(self, csp, next_state, temp_schedule=lambda i: 0):
        """ Initialize a local search instance.

        @param csp (problem): A constraint satisfaction problem instance.
        @param next_state (func): A function selecting the next state for the csp given
            the current state (assignment).
        @param temp_schedule (func): A function for calculating temperature given current
            iteration count.
        """
        self._csp = csp
        self._next_state = next_state
        self._temp = temp_schedule

    def start(self, num_iters, seed=0, rnd=False):
        """ Start the iterative local search procedure.

        @param num_iters (int): Number of iterations to run the search for.
        @param seed (float): Initial value of the random seed parameter.
        @param rnd (bool): If true, then start with a random assignment for the csp.
            Otherwise start with a greedy assignment.
        @return assignment (dict{var:val}): A complete assignment to the csp that
            satisfies all constraints. If the local search method doesn't find a solution,
            then return an empty assignment ({}).
        """
        # Initialize the csp with a complete assignment.
        random.seed(seed)
        self._csp.reset()
        if rnd: self._csp.random_assignments()
        else:   self._csp.greedy_assignments()

        # Run the iterative local search procedure.
        self._iter_count = 0
        if self._run(num_iters):
            return self._csp.getAssignments()
        return {}

    def _run(self, num_iters):
        """ Select the next state by choosing a variable and a new value to be assigned to
        that variable. If the new assignment improves the csp, then the variable is
        re-assigned, otherwise the algorithm accepts the re-assignment with some
        probability less than 1. The probability decreases exponentially with the `badness`
        of the re-assignment and the temperature.
        """
        # Iterate until a local solution is found.
        for i in range(num_iters):
            self._iter_count += 1

            # Check if the CSP is solved. If yes, then return success.
            if self._csp.isSatisfied(): return True

            # Select the next state using some strategy.
            next_state = self._next_state(self._csp)
            if next_state is None:
                return False
            var, val = next_state

            # Maybe re-assign the variable and progress to the next state.
            # Depends on the improvement and on the current temperature.
            currentVal = self._csp.getAssignments()[var]
            improvement = self._csp.n_conflicts(var, currentVal) - self._csp.n_conflicts(var, val)
            temp = self._temp(i)
            if improvement >= 0:
                self._csp.assign(var, val)
            elif temp > 0:
                p = math.exp(improvement/temp)
                if random.random() < p:
                    self._csp.assign(var, val)

        # Return failure if no solution was found for the given number of iterations.
        return False


#---------------------------------- variable selection ----------------------------------#
def maxConflictsVar(csp):
    """ Return the variable with maximum conflicts from the current assignment. Break ties
    at random.
    """
    max_conflicts = 0
    max_vars = []
    for var in csp.getVariables():
        val = csp.getAssignments()[var]
        conflicts = csp.n_conflicts(var, val)
        if conflicts > max_conflicts:
            max_conflicts = conflicts
            max_vars = [var]
        elif conflicts == max_conflicts:
            max_vars.append(var)
    return random.choice(max_vars)

def randomVar(csp):
    """ Return a random variable. """
    return random.choice(csp.getVariables())

#----------------------------------- value selection ------------------------------------#
def minConflictsVal(csp, var):
    """ Return the value that minimizes the conflicts for the given variable with respect
    to the current assignments. Break ties at random.
    """
    min_conflicts = float("inf")
    min_vals = []
    for val in csp.getDomain(var):
        conflicts = csp.n_conflicts(var, val)
        if conflicts == 0:  # short circuit value selection (sometimes improves run-time)
            return val
        if conflicts < min_conflicts:
            min_conflicts = conflicts
            min_vals = [val]
        elif conflicts == min_conflicts:
            min_vals.append(val)
    return random.choice(min_vals)

def randomVal(csp, var):
    """ Return a random value from the domain of the given variable. """
    return random.choice(list(csp.getDomain(var)))


#--------------------------------- Local search methods ---------------------------------#
def local_search(csp, num_iters, seed, rnd, next_state, temp_schedule):
    t = LocalSearch(csp, next_state, temp_schedule)
    solution = {}
    while solution == {}:
        seed = 2 * seed + seed ** 2 + 1 # pick a new seed for this trial
        solution = t.start(num_iters, seed, rnd)
    return solution

# Hill-climb search picks the best successor state from the current state.
# This is an implementation of steepest-ascent hill climbing. On every step the best
# neighbor is picked.
def hill_climb_next_state(csp):
    """ Return the best successor state. """
    best_diff = 0
    best_state = None
    assignments = csp.getAssignments()
    n_conflicts = csp.n_conflicts
    for var in csp.getVariables():
        val = assignments[var]
        current = n_conflicts(var, val)
        for v in csp.getDomain(var):
            diff = current - csp.n_conflicts(var, v)
            if diff > best_diff:
                best_diff = diff
                best_state = (var, v)
    return best_state
hill_climb = lambda csp, num_iters=10000, seed=0, rnd=False: local_search(
    csp, num_iters, seed, rnd, hill_climb_next_state, temp_schedule=lambda i: 0)


# Min-conflicts is a variant of hill-climbing search.
# Select a maximum conflicted variable.
# Select the value of that variable that minimizes its conflicts.
def min_conflicts_next_state(csp):
    """ Return a max-var min-conflict sate. """
    var = maxConflictsVar(csp)
    val = minConflictsVal(csp, var)
    return (var, val)
min_conflicts = lambda csp, num_iters=30000, seed=0, rnd=True: local_search(
    csp, num_iters, seed, rnd, min_conflicts_next_state, temp_schedule=lambda i: 0)

# Simulated annealing is a combination between hill-climbing and random walk.
# Select a random next state. If the next state is better than the current state, then it
# is accepted. Otherwise, the state is accepted with some probability less than 1.
# The probability decreases exponentially with the "badness" of the next state.
# The probability also decreases as the temperature goes down.
def simulated_annealing_next_state(csp):
    """ Return a random successor state. """
    var = randomVar(csp)
    val = randomVal(csp, var)
    return (var, val)
def temp_schedule(iter_count):
    """ Temperature goes to 0 as the iter count increases. """
    return 0.99 ** iter_count
simulated_annealing = lambda csp, num_iters=10000, seed=0, rnd=True: local_search(
    csp, num_iters, seed, rnd, simulated_annealing_next_state, temp_schedule)

#