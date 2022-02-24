from copy import deepcopy


class BacktrackingSearch:
    """ General implementation of a backtracking search algorithm.
    Each node represents a partial assignment of values to a subset of the variables and
    each child node represents an extension of the current assignment.
    With every new assignment check if the constraints are satisfied. If a constraint is
    violated, then backtrack and try a new assignment. If all constraints are satisfied
    then recurse downward by assigning an unassigned variable.
    If all variables are assigned and no constraints are violated, then we have arrived at
    a solution.
    """

    def __init__(self, csp, select_variable, ordered_domain, filter=None):
        """ Initialize a backtracking search instance.

        @param csp (problem): A constraint satisfaction problem instance.
        @param select_variable (func): A function selecting the next unassigned variable
            using some strategy.
        @param ordered_domain (func): A function giving the domain of a variable ordered
            in some way.
        @param filter (func): A function filtering the domains of the other variables
            given the current assignment.
            filter(csp, var, val) should return a list of tuples [(var1, val1), ...]
            of incompatible assignments.
        """
        self._csp = csp
        self._select_var = select_variable
        self._ordered_domain = ordered_domain
        self._filter = filter

    def _reset(self):
        # Book-keeping.
        self._num_recursive_calls = 0
        self._num_solutions_found = 0
        self._csp.reset()
        self._assignment = {}

    def start(self, all_solutions=False, verbose=False):
        """ Start the recursive backtracking search procedure.

        @param all_solutions (bool): If true, search for all solutions to the csp.
            Otherwise, return the first found solutions.
        @param verbose (bool): If true, print search statistics.
        @return assignment (dict{var:val}): A complete assignment to the csp that
            satisfies all constraints.
        """
        self._reset()
        self._recurse(all_solutions)
        if verbose:
            print(f"Executed {self._num_recursive_calls} recursive calls.")
            if all_solutions:
                print(f"Found {self._num_solutions_found} solutions to the csp.")
        return self._assignment

    def _recurse(self, all_solutions):
        """ Selects an unassigned variable and assigns it a non-conflicting value. Then
        recursively calls itself until a solution or a conflict is found.
        """
        self._num_recursive_calls += 1

        # Check if the CSP is solved.
        if self._csp.isSatisfied():
            # If we are looking for any solution, then return True.
            if not all_solutions:
                self._assignment = deepcopy(self._csp.getAssignments())
                return True
            # If we are looking for all solutions, then increment the counter and continue.
            self._num_solutions_found += 1
            self._assignment = deepcopy(self._csp.getAssignments())
            return False

        # Select the next variable using some ordering heuristic.
        var = self._select_var(self._csp)
    
        # If no variables left to consider, then return false and backtrack.
        if var is None: return False

        # Check allowable values using some ordering heuristic.
        for val in self._ordered_domain(self._csp, var):
            # If the value introduces a conflict, then skip it.
            if self._csp.n_conflicts(var, val) > 0: continue

            # Assign the value to the variable.
            self._csp.assign(var, val)

            # Modify the domains of the other variables using some filtering algorithm
            # (forward checking or Arc Consistency).
            hope, removals = self._filter(self._csp, var, val) if self._filter is not None else (True, [])

            # If any domain is empty after pruning, then continue to the next available value.
            if not hope:
                self._csp.unassign(var)
                continue

            # Prune the domains and recurse into the next variable.
            self._csp.prune(removals)
            res = self._recurse(all_solutions)

            # If the recursion was successful, then return.
            if res: return res

            # If the recursion was not successful, then restore pruned domains and continue.
            self._csp.unassign(var)
            self._csp.restore(removals)

        # If no value satisfies the constraints, then backtrack.
        return False

#---------------------------------- variable selection ----------------------------------#
def mrv(csp):
    """ Returns the variable with minimum remaining values - the variable that has the
    smallest domain.
    This function is useful when combined with a filter function to prune domains from
    inconsistent values.

    MRV can also be used in factor graphs for finding maximum weight assignments as long
    as there are some factors which are constraints (return 0 or 1).
    """
    return min(csp.getUnassigned(), key=lambda var: len(csp.getDomain(var)), default=None)

def firstUnassigned(csp):
    unassigned = csp.getUnassigned()
    if len(unassigned) > 0:
        for e in unassigned: return e
    return None

#----------------------------------- domain ordering ------------------------------------#
def lcv(csp, var):
    """ Orders the domain by the decreasing number of consistent values of neighboring
    variables.

    LCV generally cannot be used in factor graphs. It only makes sense if all factors are
    constraints. Ordering the values only makes sense if we are going to just find the
    first consistent assignment.
    """
    def sort_key(val):
        hope, removals = NC(csp, var, val)
        return len(removals) if hope else float("inf")
    return sorted(csp.getDomain(var), key=sort_key)

def unorderedDomain(csp, var):
    return csp.getDomain(var)

#----------------------------------- domain filtering -----------------------------------#
def NC(csp, var, val):
    """ A variable `var` asssigned a value `val` is node consistent with a neighbor `b` if
    all values within the domain of `b` are consistent with the constraints of the csp.
    After assigning a variable, eliminate inconsistent values from the domains of all
    unassigned neighbors of that variable. If any domain becomes empty, return.

    @param csp (satisfactionProblem): A csp instance.
    @param var (Any): CSP variable that is being assigned.
    @param val (Any): The value that is assigned to `var`.
    @return hope (bool): Return true if neighboring domains don't become empty. Otherwise
        return false - there is no hope.
    @return removals (List[(var,val)]): A list of tuples (var, val). Each tuple specifies
        a variable and the value that has to be pruned from its domain. 
    """
    assignments = csp.getAssignments()
    removals = [] # keep track of pruned values
    # Iterate over all neighbors.
    for b in csp.getNeighbors(var):
        if b in assignments or b == var:
            continue
        # Check all values from the domain of the neighbor and prune inconsistent values.
        incompatible = 0
        for bval in csp.getDomain(b):
            if not csp.constraints(var, val, b, bval):
                removals.append((b, bval))
                incompatible += 1
        # If we prune all values from the domain of a neighbor then stop the search.
        if len(csp.getDomain(b)) == incompatible:
            return False, []
    return True, removals

def AC3(csp, var, val):
    """ A variable `i` is arc consistent with respect to a variable `j` if for each value
    `v` from the domain of `i` there exists a value `w` from the domain of `j` such that
    the assignment {i:v, j:w} doesn't violate any constraints.
    Enforcing acr consistency between `i` and `j` is simply removing all values from the
    domain of `i` to make `i` arc consistent with respect to `j`.
    The arc consistency filtering algorithm enforces arc consistency on all pairs of
    variables until no domains change.
    If any domain becomes empty, return.

    @param csp (satisfactionProblem): A csp instance.
    @param var (Any): CSP variable that is being assigned.
    @param val (Any): The value that is assigned to `var`.
    @return hope (bool): Return true if neighboring domains don't become empty. Otherwise
        return false - there is no hope.
    @return removals (List[(var,val)]): A list of tuples (var, val). Each tuple specifies
        a variable and the value that has to be pruned from its domain. 
    """
    assignments = csp.getAssignments()
    removals = [] # keep track of pruned values
    # When assigning a value to `var` start by checking all arcs (b --> var), where b is
    # an unassigned neighbor to `var`.
    queue = {(b, var) for b in csp.getNeighbors(var) if b != var and b not in assignments}
    while queue:
        # Enforce arc-consistency for the arc (i --> j)
        i, j = queue.pop()
        # For every value `v` in the domain of `i` check if there exists a value `w` from
        # the domain of `j` such that the assignment {i:v, j:w} is consistent.
        maybe_remove = []
        for v in csp.getDomain(i):
            consistent = False
            for w in (csp.getDomain(j) if j != var else [val]):
                if csp.constraints(i, v, j, w):
                    consistent = True
                    break
            if not consistent:
                maybe_remove.append((i, v))
        # If we prune all values from the domain of a neighbor then stop the search.
        # Restore all pruned domains before returning.
        if len(csp.getDomain(i)) == len(maybe_remove):
            csp.restore(removals)
            return False, []
        # Prune inconsistent values from the domain of `i` if any.
        # If the domain of `i` was modified, add to the queue all arcs pointing to `i`.
        # Skip the arc (j --> i) as we know already that there exists a consistent assignment.
        if len(maybe_remove) > 0:
            csp.prune(maybe_remove)
            removals.extend(maybe_remove)
            queue.update(((k, i) for k in csp.getNeighbors(i) if k != j and k not in assignments))
    return True, removals

#------------------------------------ Abbreviations -------------------------------------#
backtrack = lambda csp, all_solutions=False, verbose=False: BacktrackingSearch(
    csp, select_variable=firstUnassigned, ordered_domain=unorderedDomain).start(all_solutions, verbose)

forward_checking = lambda csp, all_solutions=False, verbose=False: BacktrackingSearch(
    csp, select_variable=mrv, ordered_domain=lcv, filter=NC).start(all_solutions, verbose)

arc_consitency = lambda csp, all_solutions=False, verbose=True: BacktrackingSearch(
    csp, select_variable=mrv, ordered_domain=lcv, filter=AC3).start(all_solutions, verbose)

#

fwdMRV = lambda csp: BacktrackingSearch(csp, select_variable=mrv, ordered_domain=unorderedDomain, filter=NC).start(verbose=True)
arcMRV = lambda csp: BacktrackingSearch(csp, select_variable=mrv, ordered_domain=unorderedDomain, filter=AC3).start(verbose=True)
