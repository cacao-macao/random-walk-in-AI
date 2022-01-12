import time

from search.util import PriorityQueueForSMA


class Node:
    def __init__(self, item, cost, value, parent, action, depth, uuid):
        self.item = item
        self.parent = parent
        self.action = action
        self.cost = cost            # g(s)
        self.value = value          # h(s)
        self.depth = depth          # Needed to compare against the memory limit
        self.uuid = uuid            # unique identifier
        self.numChildrenInFringe = 0
        self.bestForgottenChildVal = float("-inf")
        self.successorsIterator = None # iter(problem.getSuccessors(item))
        self.allChildrenExplored = False
        # Store successor's fvalues to update parents. Use a sentinel for dead ends.
        # TODO: You should only keep track of the value of the best child in the
        # fringe and best forgotten child.
        self.successorsStorage = {"sentinel":float("inf")}

    def __eq__(self, other):
        if other is None: return False
        return self.uuid == other.uuid

    def __hash__(self):
        return hash(self.uuid)

    def decode(self):
        path = []
        curr = self
        while curr.parent is not None:
            path.append(curr.action)
            curr = curr.parent
        path.reverse()
        return path


def smaStar(problem, heuristic, limit=1_000_000):

    def update(node):
        if node.allChildrenExplored:
            if i % print_every == 0:
                print("    updating node:", node.item)
            oldValue = node.value
            node.value = max(node.value, min(list(node.successorsStorage.values())))
            if oldValue != node.value:
                if i % print_every == 0:
                    print("    updating node priority:", (node.value, -node.depth))
                oldLen = len(fringe)
                fringe.update(node, (node.value, -node.depth, -node.uuid))
                assert oldLen == len(fringe)
                if node.parent != None:
                    node.parent.successorsStorage[node.item] = node.value
                    update(node.parent)

    print_every = 100_000
    uuid = 0
    fringe = PriorityQueueForSMA()
    leaves = PriorityQueueForSMA()
    startState = problem.getStartState()
    startNode = Node(startState, 0, heuristic(startState, problem.goal), None, None, 0, uuid)
    fringe.push(startNode, (0+heuristic(startState, problem.goal), 0, -uuid))
    leaves.push(startNode, (0-heuristic(startState, problem.goal), 0, uuid))
    uuid += 1

    tic = time.time()
    i = 0
    while not fringe.isEmpty():
        # _ = input()

        # Get the top node from the fringe
        #
        p, idx, curr = fringe.peek()    # Get the oldest deepest least-fval node (this node is always a leaf!)

        i += 1
        if i % print_every == 0:
            toc = time.time()
            print(f"iter: {i}\tlen(fringe): {len(fringe)}\tbest node: {curr.value}\t{print_every} iterations took {toc-tic:4f} seconds")
            print(f"best: {curr.item}\tpriority:{p}\tindex: {idx}")
            tic = time.time()

        # Halting test. If the best node has priority infinity, then give up.
        #
        if p[0] == float("inf"):
            print("Iterations:", i)
            print("Memory limit is not enough to find the solution. The solution is deeper.")
            print(f"Leaves: {len(leaves)}\tFringe: {len(fringe)}")
            return []

        # Goal test.
        #
        if problem.isGoalState(curr.item):
            return curr.decode()


        # Generate the next successor.
        #
        # This should generate the best forgotten child if any.
        if curr.successorsIterator == None: # This may be moved inside Node __init__ func
            curr.successorsIterator = iter(problem.getSuccessors(curr.item))
        try:
            s, a, c = next(curr.successorsIterator)
            fval = max(curr.value, curr.cost + c + heuristic(s, problem.goal))
            if i % print_every == 0:
                print("  next successor available:", s)

            _curr, flag = curr, False
            while _curr is not None:
                if s == _curr.item: flag = True; break
                _curr = _curr.parent
            if flag:
                fval = float("inf")
                if i % print_every == 0:
                    print("  skipping child because it creates a cycle:", s)
                continue

            depth = curr.depth + 1
            if depth >= limit:
                fval = float("inf")
                if i % print_every == 0:
                    print("  skipping child because its too deep:", s)
                continue


        # If all successors are generated, then update the fvalue of the node.
        #
        except StopIteration:   # Update the parent value and start-over
            if i % print_every == 0:
                print("  completed iterating successors")
            curr.allChildrenExplored = True
            update(curr)

            if curr in leaves:
                leaves.update(curr, (-curr.value, curr.depth, curr.uuid))
            curr.successorsIterator = None  # This is not right!
            # TODO: successorsIterator should contain only the children that are not in
            # the fringe.
            continue


        # Remove the parent from the leaves structure. This is necessary because the
        # parent might be the worst leaf and if the fringe is full it will be removed.
        # There might be worse nodes, but if this is the first expanded child, then the
        # parent might be the worst leaf node.
        #
        curr.successorsStorage[s] = fval
        curr.numChildrenInFringe += 1
        leaves.remove(curr)
        if i % print_every == 0:
            print(f"  successor fval = max({curr.value}, {curr.cost} + {c} + {heuristic(s, problem.goal)} = {fval})")
            print(f"  removing from leaves: {curr.item}")

        # TODO:
        # if all successors in memory then remove curr from fringe


        # If the memory is full, then remove the worst leaf from the fringe.
        #
        if len(fringe) == limit:
            discardedNode = leaves.pop()
            fringe.remove(discardedNode)
            discardedNode.parent.numChildrenInFringe -= 1

            if i % print_every == 0:
                print(f"  dropping worst leaf with value {discardedNode.value} and depth {discardedNode.depth}: {discardedNode.item}")
                print(f"  parent num children in fringe is {discardedNode.parent.numChildrenInFringe}")
                if discardedNode.parent.numChildrenInFringe == 0: print(f"  pushing parent to leaves")
    
            if discardedNode.parent.numChildrenInFringe == 0:
                leaves.push(discardedNode.parent, (-discardedNode.parent.value, discardedNode.parent.depth, discardedNode.parent.uuid))

            # # TODO: Update the parent fvalue for the discared node.
            # discardedNode.parent.bestFrogottenChildVal = min(
            #     discardedNode.parent.bestFrogottenChildVal, discardedNode.value)
            # # maybe inset discardedNode.parent to the queue
            #     # TODO: If removing curr when all successors in memory, then check if curr
            #     # is in the fringe
            #     if discardedNode.parent not in fringe:
            #         fringe.push(discardedNode.parent, discardedNode.parent.value)


        # Insert the new node to the fringe.
        #
        child = Node(s, curr.cost+c, fval, curr, a, depth, uuid)
        uuid += 1
        fringe.push(child, (fval, -depth, -uuid))
        leaves.push(child, (-fval, depth, uuid))
        if i % print_every == 0:
            print("  adding child: ", s)
            print("  child priority: ", (fval, -depth, -uuid))

#