from collections import defaultdict

from search.util import LimitedDict, PriorityQueue, Queue, Stack


class TreeSearch:
    """ General implementation of a tree search algorithm.
    Searches for a solution to the given problem by pop-ing items from the fringe and
    iteratively exploring the successors of the pop-ed item.
    """

    #-------------------------------- nested Node class ---------------------------------#
    class Node:
        """ Lightweight, nonpublic class for storing a tree node. """

        # Streamline memory usage.
        __slots__ = "_item", "_parent", "_action", "_cost"

        def __init__(self, item, parent, action, cost):
            """ Initialize a Node instance.

            @param item (searchState): The state from the search problem to which this
                node corresponds.
            @param parent (Node): The node in the tree that generated this node.
            @param action (searchAction): The action that was applied to the parent's
                state to generate this node's state.
            @param cost (float): The total cost of the path from the initial state to this
                node.
            """
            self._item = item
            self._parent = parent
            self._action = action
            self._cost = cost

        @property
        def item(self): return self._item
        @property
        def parent(self): return self._parent
        @property
        def action(self): return self._action
        @property
        def cost(self): return self._cost
        @parent.setter
        def parent(self, newParent): self._parent = newParent
        @action.setter
        def action(self, newAction): self._action = newAction
        @cost.setter
        def cost(self, newCost):  self._cost = newCost

        def __hash__(self): return hash(self.item)

    #------------------------------ container initializer -------------------------------#
    def __init__(self, fringe):
        """ Initialize a tree search instance.
        The tree search objects takes a fringe class constructor. Example usage:
            ```
            class Stack:
                def __init__(self): ...
                def __len__(self): ...
                def push(self, item): ...
                def pop(self): ...

            dfs = TreeSearch(Stack)
            solution = dfs.start(problem)
            ```

        @param fringe (Fringe object): A class constructor for the data structure used for
            the fringe. The fringe must support push(item, [priority]), pop() and len()
            methods.
        """
        self._fringe = fringe()

        # Check how many arguments fringe.push() takes and crate a wrapper function
        # around it that can be called with both 1 and 2 arguments.
        # self._fringePush = self._fringe.push
        if self._fringe.push.__code__.co_argcount == 3:     # push(self, item, priority)
            self._fringePush = lambda item, priority: self._fringe.push(item, priority)
        elif self._fringe.push.__code__.co_argcount == 2:   # push(self, item) - Stack, Queue
            self._fringePush = lambda item, priority=0: self._fringe.push(item)
        else:
            raise SyntaxError("expected fringe.push(self, item, [priority]) method")

    #--------------------------------- public accessors ---------------------------------#
    def start(self, problem, heuristic=None, track=True, limit=float("inf"), verbose=False):
        """ Run the tree search algorithm and build the solution path.

        @param problem (searchProblem): A search problem object instance.
        @param heuristic (func): A function from a search state to a non-negative number.
        @param track (bool): If true; track visited states using a visited set.
        @param limit (float): Limiting value for searching.
        @param verbose (bool): If true, print search statistics.
        @return path (List[actions]): A list of actions giving the path from the start
            state to the solution state.
        """
        goal, _ = self._run(problem, heuristic, track, limit, verbose)
        return self._decode(goal)

    #--------------------------------- private methods ----------------------------------#
    def _run(self, problem, heuristic, track, limit, verbose):
        """ Start the tree search and continuously explore the state space until a
        solution is found. Visited nodes may be tracked to reduce redoing some work,
        however tracking nodes necessitates using larger amounts of memory as search
        progresses.
        If a limit is set, then reduce the search tree by pruning branches that exceed
        the limit.
        Return the goal node and the minimum cost that exceeded the limit (-1 if no limit).

        @param problem (searchProblem): A search problem object instance.
        @param heuristic (func): A function from a search state to a non-negative number.
        @param track (bool): If true; track visited states using a visited set.
        @param limit (float): Limiting value for searching.
        @param verbose (bool): If true, print search statistics.
        @return goal (Node): Return the goal node representing the solution to the problem.
        @return epsi_limit (float): Minimum state value that exceeded the limit.
        """
        # explored is the number of currently explored states
        explored = 0

        # max_nodes is the maximum number of nodes that can be stored in the fringe or
        # in the visited set before memory overflow occurs
        # If the visited set reaches the limit, then newly visited nodes will not be stored.
        # if the fringe reaches the limit, then the procedure stops.
        max_nodes = 3_000_000
        visited = LimitedDict(max_nodes) # visited states dictionary {state : cost}

        # If no heuristic, then use trivial zero-heuristic. Reduces if-else cases.
        heuristic = (lambda state, goal: 0) if heuristic is None else heuristic
        epsi_limit = float("inf") # store the minimum state value that exceeds the limit

        # Add the start node to the fringe and iteratively explore its successors.
        initial = problem.getStartState()
        visited[initial] = 0
        startNode = self.Node(parent=None, item=initial, action=None, cost=0)
        self._fringePush(startNode, 0)

        while not self._fringe.isEmpty():
            node = self._fringe.pop()
            explored += 1
            if verbose and explored % 100000 == 0:
                print(f"explored {explored} nodes")

            # Check if the current node is a solution to the problem.
            if problem.isGoalState(node.item):
                return node, -1

            # Expand te current node and explore its successors.
            for s, a, c in problem.getSuccessors(node.item):
                ##########################################################################
                # TODO: Use this with a boolean parameter use_g to switch from A* to Greedy 
                # cost_until_now = node.cost + c if use_g else 0
                # cost_to_go = heuristic(s, problem.goal)
                # cost = cost_until_now + cost_to_go
                ##########################################################################
                cost = node.cost + c + heuristic(s, problem.goal)
                child = self.Node(item=s, parent=node, action=a, cost=node.cost+c)

                # Maybe limit the depth of the search.
                if cost > limit:
                    epsi_limit = min(epsi_limit, cost)
                    continue

                # Maybe track the visited states.
                if track:
                    if s in visited and visited[s] <= node.cost + c:
                        continue
                    else:
                        visited[s] = child.cost # visited is a dict limited by `max_nodes`

                # Check for cycles in the chain of parents.
                curr, flag = node, False
                while curr is not None:
                    if s == curr.item:
                        flag = True
                        break
                    curr = curr.parent
                if flag:
                    continue

                # If child is a new state, then push to the fringe.
                self._fringePush(child, cost)

                # Memory limits could halt the procedure.
                if len(self._fringe) > max_nodes:
                    print("The fringe exceeded the memory limit... :/")
                    print(f"  fringe has {len(self._fringe)} nodes")
                    return None, -1

        return None, epsi_limit

    def _decode(self, goalNode):
        """ Given a goalNode produced from the tree search algorithm, trace the backward
        path from the goal state to the start state by following the pointers to the
        parents. Return the reversed path.

        @param goalNode (Node): Node object returned from the tree search algorithm.
        @return path (List[action]): A list of actions giving the path from the start
            state to the goalNode state.
        """
        if goalNode is None: return []
        path = []
        curr = goalNode
        while curr.parent is not None:
            path.append(curr.action)
            curr = curr.parent
        path.reverse()
        return path


#------------------------------------ Abbreviations -------------------------------------#
nullHeuristic = lambda state, goal=None: 0

dfs = lambda problem, track=True, verbose=False: TreeSearch(Stack).start(problem, track=track, verbose=verbose)
bfs = lambda problem, track=True, verbose=False: TreeSearch(Queue).start(problem, track=track, verbose=verbose)
ucs = lambda problem, track=True, verbose=False: TreeSearch(PriorityQueue).start(problem, nullHeuristic, track, verbose=verbose)
astar = lambda problem, heuristic=nullHeuristic, track=True, verbose=False: \
            TreeSearch(PriorityQueue).start(problem, heuristic, track, verbose=verbose)

def ids(problem, heuristic=None, track=True, verbose=True):
    def dls(limit):
        t = TreeSearch(Stack)
        goal, lim = t._run(problem, heuristic, track, limit, verbose)
        path = t._decode(goal)
        return path, lim
    path, limit = [], 1
    while len(path)==0:
        if verbose: print("exploring limit:", limit)
        path, limit = dls(limit)
    return path  


def rbfs(problem, heuristic=lambda state, goal: 0, verbose=False):

    def recurse(node, limit, visited):
        F = defaultdict(lambda item: 0)
        state, cost, fval = node
        if state in visited: return None, float("inf"), []
        visited.add(state)
        if problem.isGoalState(state): return node, -1, []
        successors = problem.getSuccessors(state)
        for s, a, c in successors:
            F[s] = max(cost + c + heuristic(s, problem.goal), fval)

        while True:
            best, a, c = min(successors, key=lambda succ: F[succ[0]])
            if F[best] > limit:
                return None, F[best], []
            succsWObest = successors[:]
            succsWObest.remove((best, a, c))
            if len(succsWObest) > 0:
                secondBest, _, _ = min(succsWObest, key=lambda succ: F[succ[0]])
                alternative = F[secondBest]
            else:
                alternative = float("inf")
            child = (best, cost + c, F[best])
            res, new_val, path = recurse(child, min(limit, alternative), visited.copy())
            F[best] = new_val
            if res != None:
                return res, -1, path + [a]

    start = (problem.getStartState(), 0, 0)
    visited = set()
    _, _, path = recurse(start, float("inf"), visited)
    path.reverse()
    return path

#


"""
##########################################################################################

Notes on revisiting states when searching graphs.
Search algorithms: dfs, bfs, ucs, a*, dls, ids, rbfs, sma*

  - If the state space is finite and fits into memory then we can use a visited set to
keep track of visited nodes.
  - If the problem is such that it is rare or impossible to reach the same state following
different paths, then we don't need a visited set.
  - If the state space is infinite and state repetition is possible, then we could at best
check for cycles.
    (Russel, Norvig: Some implementations follow the chain of parent pointers all the way
    up, and thus eliminate all cycles; other implementations follow only a few links.)
    (In the case of DFS we store the currently explored path.)


==========================================================================================
==========================================================================================
First, consider the case where we could afford to keep track of all visited nodes.
Two approaches have to be considered when keeping track of visited states:
    (using a set for visited states)
    - Performing the check before expanding the state, after poping from the fringe
        ```
        while fringe.notEmpty():
            node = fringe.pop()
            state = node.item
            if state in visited: continue
            visited.add(state)
            ...
        ```
    - Performing the check on the successors, before pushing to the fringe
        ```
        while fringe.notEmpty():
            ...
            for s, a, c in successors:
                if s in visited: continue
                visited.add(s)
                ...
        ```
If the first approach is used, then the fringe utilizes more memory because it keeps
possibly duplicated states.

------------------------------------------------------------------------------------------
For the simple depth first search and breadth first search the second approach is much
better.

------------------------------------------------------------------------------------------
For uniform cost search and A* search the second approach needs to be modified in order to
work. The first approach, while using more memory, works out of the box and needs no
comment.
To see why the second approach needs fixing consider the following: If a state was visited
but a cheaper path has been found, then the cost of that state must be modified both in
the visited structure and in the fringe. We must also not forget to modify the parent and
action fields. The `update` method updates the priority of the state if it is still in the
fringe, or adds it to the fringe if it has been pop-ed.
    (using a dict for visited states: visited = {state : node})
        ```
        while fringe.notEmpty():
            ...
            for s, a, c in successors:
                if s in visited and visited[s].cost < parent.cost + c:
                    continue
                visited[s].parent = parent
                visited[s].action = a
                visited[s].cost = parent.cost + c
                fringe.update(visited[s], parent.cost + c + heuristic(s))
                ...                                        (in case of A*)
        ```
    It should be noted that `update` must be optimized. Simple priority queues implemented
    using a min heap need O(n) time for update, because there is no way to find the entry
    in the heap fast.
    Smarter implementations could keep a mapping from entries to heap positions. A
    workaround for this problem would be to simply add the successor again to the fringe.
    Adding to a min heap takes O(logn) time and pushes the new node above the old one. The
    new node will be expanded first and when the old node is expanded all of its
    successors will already have been visited with lower cost. However, if memory usage is
    a problem, then the approach with a better implementation for the fringe should be
    used.
    (using a dict for visited states: visited = {state : cost})
        ```
        while fringe.notEmpty():
            ...
            for s, a, c in successors:
                if s in visited and visited[s] < parent.cost + c:
                    continue
                visited[s] = parent.cost + c
                fringe.push(s, parent.cost + c + heuristic(s))  # the fringe is a min heap
                ...                             (in case of A*)
        ```
    It can be argued whether using a better implementation for the fringe or using the
    workaround is better as there is another more subtle tradeoff. In the first case the
    visited set must be a mapping from states to tree nodes, while in the second case the
    mappings is more lightweight {state: cost}.

------------------------------------------------------------------------------------------
For depth limited search, again, the first approach works out of the box.
    ```
    while fringe.notEmpty():
        node = fringe.pop()
        state = node.item
        if cost(state) > limit: continue    # check the limit before checking if visited !!!
        if state in visited: continue
        visited.add(state)
        ...
    ```
    For the second approach, adding child nodes directly to the visited set would prevent
    us from finding a solution even if one exists and is shorter than the limit. Thus, we
    need to make an adjustment similar to the fix for UCS and A*.
    Consider the example below: searching depth first the algorithm would expand the
    following nodes:
        S -> A - depth 1
        S -> B - depth 1
        A -> C - depth 2
        A -> D - depth 2
        D -> E - depth 3
        D -> F - depth 3 --- adds F to the visited set
        B -> F - depth 2 --- already visited - END!
    The path to the goal is 3 steps long but depth limited search could not find it even
    though the limit was 3 steps.
                    S
                  /   \
                 A     B
               /   \    \
              C     D    \
                  /   \  /
    ----------   E     F    ----------- depth limit = 3 -----------------
                     /   \
                    G    GOAL
    To address this problem we could use the second modification described for UCS and A*
    (the first modification reduces to the second when fhe fringe is a stack).
    If a state was visited but a cheaper path has been found, then the cost of that state
    must be modified in the visited structure and it must be added to the fringe.
    With this approach nodes will be visited multiple times.
    (using a dict for visited states: visited = {state : cost})
        ```
        while fringe.notEmpty():
            ...
            for s, a, c in successors:
                if s in visited and visited[s] < parent.cost + c:
                    continue
                visited[s] = parent.cost + c
                fringe.push(s)  # the fringe is a stack
                ...
        ```

------------------------------------------------------------------------------------------
Iterative deepening uses depth limited search as a subroutine so it needs no comment.

==========================================================================================
==========================================================================================
When considering infinite graphs, however, memory usage of the visited set becomes very
expensive.

Keeping track of the entire set of visited states while traversing infinite cyclic graphs
is actually not necessary, but if we could afford the memory it could speed up the search
massively.
When running depth first search to avoid cycles we must keep track of the current path we
are traversing, but this is no concern.
Breadth first search, Uniform cost search, A*, Depth limited and Iterative deepening run
perfectly normal even without keeping track of visited nodes.
We have the following theoretical complexities for these algorithms (without visited set):

    Algorithm   Running Time    Fringe Memory
    -----------------------------------------
    DFS         O(b^d)          O(bd)
    BFS         O(b^d)          O(b^d)
    UCS         O(b^(C*/epsi))  O(b^(C*/epsi))      (C* - cost of solution; epsi - min cost of an edge)
    A*          O(b^d)          O(b^d)
    IDS         O(b^d)          O(bd)
    -----------------------------------------

where b is the branching factor and d is the depth of the best solution to the problem.
When using a visited set the memory complexity for any algorithm is O(|V|).
However, when a visited set is not used, using BFS, UCS and A* is practically impossible.
And using DFS in general on infinite graphs would never find the best solution.
Iterative deepening is probably the best way to handle the problem of infinite state
spaces.

For the simple depth first search the following would do:

        ```
        current_path = []   # current path keeps (immutable) states, not nodes !
        while fringe.notEmpty():
            node = fringe.pop()
            state = node.item
            # if state in current_path: continue -----------------------------
                                                                             |
            # pop states from the current path until the parent              |
            # of the currently explored node is reached                      |
            while node.parent.item != current_path[-1]:                      |
                current_path.pop()                                           |---> again two cases
            current_path.append(state)                                       |
            ...                                                              |
                                                                             |
            # for s, a, c in successors: ------------------------------------|
            #     if s in current_path: continue ----------------------------|
            #     fringe.push(child) ----------------------------------------|
            ...
        ```
Using this approach utilizes a lot les memory, but states will be visited multiple times,
thus increasing dramatically the run-time of the procedure. The search will revisit nodes
exponentially many times which will slow their running time, but the theoretical
worst-case bounds remain the same.

Even using A* search with an admissible and consistent heuristic would revisit states
exponentially many times. For example consider the case where an action is reversible.
Then start from the start state S and take an action leading to state A. After exploring
the children of A we will add S to the fringe again (because S will be in the successors
list). The new priority of S will be old_priority + cost(S->A) + cost(A->S), which means
that if we store a visited set we would not consider this state. However, when we are not
storing a visited set this copy of S will be revisited when all the nodes in the fringe
with lower priority are explored, and we now this will happen because successor heuristics
are at best non-decreasing (h(S) <= cost(S->A) + h(A))


IDA*, RBFS, SMA* ?

"""

#