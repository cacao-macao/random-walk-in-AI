import time


class SatisfactionAgent:
    """ General implementation of a satisfaction agent.
    Uses the given satisfaction method to search for a solution to the given problem.
    """

    def __init__(self, problem, method):
        """ Initialize a satisfaction agent instance.

        @param problem (satisfactionProblem): A satisfaction problem class.
        @param method (func): A satisfaction method function.
        """
        self.problem = problem
        self.method = method

    def registerCSP(self, variables, domains, neighbors, constraints):
        """ This is the first time that the agent sees the layout of the csp. Here, we
        choose a complete and consistend assignment. In this phase, the agent should find
        the solution and store it in a local variable.
        All of the work is done in this method!

        @param variables (List): A list of variables for the CSP.
        @param domains (dict): A dictionary giving the domain of values for every variable.
        @param neighbors (dict): A dictionary giving the list of neighbors for every
            variable.
        @param constraints (func): A binary constraint function returning true if the
            assignment {var1:val1, var2:val2} is consistent.
        """
        tic = time.time()
        problem = self.problem(variables, domains, neighbors, constraints)
        self.assignments = self.method(problem)
        toc = time.time()
        if len(self.assignments) == 0:
            print("Could not find a solution")
        else:
            print(f"Consistent assignment found in {toc-tic:.4f} seconds")

#