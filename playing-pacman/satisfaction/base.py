import random


class BaseSatisfactionProblem:
    """ An implementation of a (finite-domain) constraint satisfaction problem.

    A CSP is specified by the following data structures:
        variables:      A list of variables; each is atomic (e.g. int or string).
        domains:        A dict of {var : {possible_value, ...}} entries.
        neighbors:      A dict of {var : [var,...]} that for each variable lists
                        the other variables that participate in constraints.
        constraints:    A function f(A, a, B, b) that returns true if neighbors
                        A, B satisfy the constraint when they have values A=a, B=b.
        assignments:    A dict of {var : val} that stores currently assigned variables
                        and their assignments.
        unassigned:     A set of variables that haven't been assigned a value yet.

        # Not implemented
        # conflicted:     A set of variables that have conflicting assignments.

    A satisfaction problem defines the basic actions you can perform: assign or unassign a
    variable; measure the conflicts assigning a new variable introduces; check if the
    current assignment is complete and satisfies all constraints.
    """

    def __init__(self, variables, domains, neighbors, constraints):
        """
        Initialize a basic CSP instance.

        @param variables (List): A list of variables for the CSP.
        @param domains (dict): A dictionary giving the domain of values for every variable.
            Use a set for the domain to facilitate fast pruning and restoring.
        @param neighbors (dict): A dictionary giving the list of neighbors for every
            variable.
        @param constraints (func): A binary constraint function returning true if the
            assignment {var1:val1, var2:val2} is consistent.
        """
        self._variables = variables
        self._domains = domains
        self._neighbors = neighbors
        self.constraints = constraints
        self.reset()

    def reset(self):
        """ Reset the state of the csp. """
        # The current state of the CSP is represented by the assignments of the variables.
        self._assignments = {}
        # Keep track of unasigned variables.
        self._unassigned = set(self._variables)

    def assign(self, var, val):
        """ Add {var: val} to the assignments. Discard the old value if any. """
        if var in self._unassigned:
            self._unassigned.discard(var)
        self._assignments[var] = val

    def unassign(self, var):
        """ Remove {var: val} from the assignment. Do not call this if for changing a
        variable to a new value; just call `assign` for that.
        """
        if var in self._assignments:
            self._unassigned.add(var)
            del self._assignments[var]

    def n_conflicts(self, var, val):
        """ Return the number of conflicts the variable `var` has with other assigned
        variables.
        """
        conflict = lambda var2: var2 in self._assignments and \
                not self.constraints(var, val, var2, self._assignments[var2])
        return sum(conflict(v) for v in self._neighbors[var])

    def isSatisfied(self):
        """ Return true if the assignment is complete and satisfies all the constraints. """
        return len(self._assignments)==len(self._variables) and all(
            self.n_conflicts(var, self._assignments[var]) == 0 for var in self._variables)

    def getVariables(self):
        """ Return the list of variables. """
        return self._variables

    def getAssignments(self):
        """ Return the current state of the assignments. """
        return self._assignments

    def getDomain(self, var):
        """ Return the domain for the given variable. """
        return self._domains[var]
    
    def getNeighbors(self, var):
        """ Return the list of neighbors for the given variable. """
        return self._neighbors[var]
    
    def getUnassigned(self):
        """ Return an iterable of the unassigned variables (if any). """
        return self._unassigned

    def getConflicted(self):
        """ Return an iterable of the conflicted variables (if any). """
        return self._conflicted

    def prune(self, removals):
        """ Prune the domains of all variables from the removals list.

        @param removals (List[(var, val)]): A list of tuples containing pairs (var, val)
            giving values `val` to be pruned from the domain of variables `var`.
        """
        for var, val in removals:
            self._domains[var].discard(val)

    def restore(self, removals):
        """ Restore the domains of all variables after pruning.

        @param removals (List[(var, val)]): A list of tuples containing pairs (var, val)
            giving values `val` pruned from the domain of variables `var`.
        """
        for B, b in removals:
            self._domains[B].add(b)

    def greedy_assignments(self):
        """ Assign all variables of the csp using a greedy algorithm. For every variable
        choose a value that results in 0 conflicts if possible. Skip variables that cannot be
        assigned a non-conflicting value. In the end assign all skipped values at random.
        """
        self.reset()
        variables = list(self.getVariables())
        random.shuffle(variables)
        # Iterate over all variables and assign a value with 0 conflicts (if possible).
        for var in variables:
            # Search for a value that produces no conflicts.
            for val in self.getDomain(var):
                if self.n_conflicts(var, val) == 0:
                    self.assign(var, val)
                    break
        # Assign all other variables randomly.
        for var in list(self.getUnassigned()):                  # copy the set of unassigned variables
            val = random.choice(list(self.getDomain(var)))      # to avoid runtime error
            self.assign(var, val)                               # 'Set changed size during iteration'

    def random_assignments(self):
        """ Generate a complete assignment by assign all variables of the csp at random. """
        self.reset()
        for var in self._variables:
            val = random.choice(list(self._domains[var]))
            self.assign(var, val)

#