import random

from satisfaction.base import BaseSatisfactionProblem


class NQueensSatisfaction(BaseSatisfactionProblem):
    """ This satisfaction problem is used to find solution states to the N-Queens problem.
    States are represented as partial assignments of queens to board positions.
    We have N queens on an N x N board. We must have one queen in every column. If we
    think of placing queens one per column, from left to right, then this means that
    position (x,y) represents (var, val) in the CSP.
    """

    def reset(self):
        """ Overwrite the method to keep track of conflicts in each row, down diagonal and
        up diagonal.
        Take advantage of the fact that all variables have the same domain. To reduce
        memory usage use a single set to store the domain.
        """
        self.N = len(self._variables)

        # The values are the row positions each queen can occupy.
        domain = set(range(self.N))
        self._domains = {var : domain for var in self._variables}

        # The current state of the CSP is represented by the assignments of the variables.
        self._assignments = {}

        # Keep track of unasigned variables.
        self._unassigned = set(self._variables)

        # Three arrays will be used to keep track of the count of queen conflicts.
        # rows[i]: Number of queens in the ith row
        # downs[i]: Number of queens in the \ diagonal such that their (x,y) coordinates sum to i
        # ups[i]: Number of queens in the / diagonal such that their (x,y) coordinates have x-y+n-1 = i
        self._rows = [0] * self.N
        self._downs = [0] * (2 * self.N -1)
        self._ups = [0] * (2 * self.N -1)

        self._separate_domains = set()

    def assign(self, var, val):
        """ Overwrite the method to keep track of conflicts in each row, down diagonal and
        up diagonal.
        """
        oldVal = None
        if var in self._unassigned:
            self._unassigned.discard(var)
        else:
            oldVal = self._assignments[var]
            if oldVal == val: return
            self._record_conflict(var, oldVal, -1)
        self._assignments[var] = val
        self._record_conflict(var, val, 1)

    def unassign(self, var):
        """ Overwrite the method to keep track of conflicts in each row, down diagonal and
        up diagonal.
        """
        if var in self._assignments:
            oldVal = self._assignments[var]
            self._record_conflict(var, oldVal, -1)
            self._unassigned.add(var)
            del self._assignments[var]

    def _record_conflict(self, var, val, delta):
        """ Record conflicts caused by addition or deletion of a Queen. """
        n = len(self._variables)
        self._rows[val] += delta
        self._downs[var+val] += delta
        self._ups[var-val+n-1] += delta

    def n_conflicts(self, var, val):
        """ Overwrite the method to make use of the rows, downs and ups lists. Counting
        the number of conflicts is done in O(1) time.
        Count the number of conflicts for the variable `var` by counting the number of
        queens residing in each row, down diagonal and up diagonal (we have one queen per
        column!). If the variable has already been assigned this value, then subtract 3
        because of double counting (queens cannot conflict with themselves).
        """
        n = self.N
        c = self._rows[val] + self._downs[var+val] + self._ups[var-val+n-1]
        if self._assignments.get(var, None) == val:
            c -= 3
        return c         

    def greedy_assignments(self):
        """ Overwrite the method to make use of the fact that the CSP is fully connected,
        i.e. every variable neighbors all other variables. This means that every variable
        must be assigned a different value from the domain to satisfy the constraints.
        While greedily assigning variables, prune the domain from values that have already
        been assigned. This will speed up the initialization process because values that
        are certain to fail will not be checked.
        """
        N = self.N
        variables = list(self.getVariables())
        random.shuffle(variables)
        domain = set(range(N))
        # Iterate over all variables and assign a value with 0 conflicts (if possible).
        for var in variables:
            # Search for a value that produces no conflicts.
            for val in domain:
                if self.n_conflicts(var, val) == 0:
                    self.assign(var, val)
                    domain.remove(val) # if a value is assigned, remove it from the domain
                    break
        # Assign all other variables randomly.
        domain = list(range(N))
        random.shuffle(domain)
        for var in list(self.getUnassigned()):
            val = domain.pop()
            self.assign(var, val)

    def prune(self, removals):
        """ Overwrite the base method to take into consideration the fact that all
        variables point to the same domain list.
        For memory efficiency considerations all the variables share the same domain set.
        In order to prune the domain of a variable, first the domain must be copied in a
        different container.

        @param removals (List[(var, val)]): A list of tuples containing pairs (var, val)
            giving values `val` to be pruned from the domain of variables `var`.
        """
        for var, val in removals:
            if var not in self._separate_domains:
                self._domains[var] = set(self._domains[var])
                self._separate_domains.add(var)
            self._domains[var].discard(val)

#

def nonatacking_pairs(assignments, constraints):
    count = 0
    N = len(assignments)
    for i in range(N-1):
        for j in range(i+1, N):
            v = assignments[i]
            w = assignments[j]
            if constraints(i, v, j, w):
                count += 1
    return count

def printBoard(nqueens):
    N = nqueens.N
    rows = [[] for _ in range(N)]
    for col, row in nqueens.getAssignments().items():
        rows[row].append(col)
    lines = []
    horizontalLine = '-' * (4*N+1)
    print(rows)
    for qs in rows:
        lines.append(horizontalLine)
        if len(qs) == 0:
            lines.append("|   "*N + "|")
        else:
            line = []
            prev = -1
            for q in sorted(qs):
                line.append("|   "*(q-prev-1) + "| * ")
                prev = q
            line.append("|   "*(N-prev-1) + "|")
            lines.append(''.join(line))
    lines.append(horizontalLine)
    print("\n".join(lines))


def printNQueens(assignments):
    N = len(assignments)
    rows = [[] for _ in range(N)]
    for col, row in assignments.items():
        rows[row].append(col)
    lines = []
    horizontalLine = '-' * (4*N+1)
    for qs in rows:
        lines.append(horizontalLine)
        if len(qs) == 0:
            lines.append("|   "*N + "|")
        else:
            line = []
            prev = -1
            for q in sorted(qs):
                line.append("|   "*(q-prev-1) + "| * ")
                prev = q
            line.append("|   "*(N-prev-1) + "|")
            lines.append(''.join(line))
    lines.append(horizontalLine)
    print("\n".join(lines))

#