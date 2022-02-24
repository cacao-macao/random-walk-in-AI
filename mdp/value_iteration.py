from itertools import cycle

import util

def full_value_iteration(mdp, discount, num_iters):
    values = {s: 0 for s in mdp.states()}
    for _ in range(num_iters):
        new_values = values.copy()
        for s in mdp.states():
            if mdp.is_terminal(s):
                new_values[s] = 0
                continue
            max_val = float("-inf")
            for a in mdp.actions(s):
                new_val = 0
                for next_s, prob in mdp.transitions_and_probs(s, a):
                    new_val += prob * (mdp.reward(s, a, next_s) + discount*values[next_s])
                max_val = max(max_val, new_val)
            new_values[s] = max_val
        values = new_values.copy()
    return values


def asynchronous_value_iteration(mdp, discount, num_iters):
    states_cycle = cycle(mdp.states())
    values = {s: 0 for s in mdp.states()}
    for _ in range(num_iters):
        s = next(states_cycle)
        if mdp.is_terminal(s):
            values[s] = 0
            continue
        max_val = float("-inf")
        for a in mdp.actions(s):
            new_val = 0
            for next_s, prob in mdp.transitions_and_probs(s, a):
                new_val += prob * (mdp.reward(s, a, next_s) + discount*values[next_s])
            max_val = max(max_val, new_val)
        values[s] = max_val
    return values


def priority_sweep_value_iteration(mdp, discount, num_iters, theta=1e-5):
    prioritySweep = util.PriorityQueue()
    values = {s : 0 for s in mdp.states()}
    predecessors = {s : set() for s in mdp.states()}
    for s in predecessors.keys():
        for a in mdp.actions(s):
            for next_s, _ in mdp.transitions_and_probs(s, a):
                predecessors[next_s].add(s)

    for s in mdp.states():
        if mdp.is_terminal(s):
            continue
        max_val = float("-inf")
        for a in mdp.actions(s):
            new_val = 0
            for next_s, prob in mdp.transitions_and_probs(s, a):
                new_val += prob * (mdp.reward(s, a, next_s) + discount*values[next_s])
            max_val = max(max_val, new_val)
        diff = abs(max_val - values[s])
        prioritySweep.push(s, -diff)

    for _ in range(num_iters):
        if prioritySweep.isEmpty():
            break
        s = prioritySweep.pop()
        max_val = float("-inf")
        for a in mdp.actions(s):
            new_val = 0
            for next_s, prob in mdp.transitions_and_probs(s, a):
                new_val += prob * (mdp.reward(s, a, next_s) + discount*values[next_s])
            max_val = max(max_val, new_val)
        values[s] = max_val

        for p in predecessors[s]:
            max_val = float("-inf")
            for a in mdp.actions(p):
                new_val = 0
                for next_s, prob in mdp.transitions_and_probs(p,a):
                    new_val += prob * (mdp.reward(p, a, next_s) + discount*values[next_s])
                max_val = max(max_val, new_val)
            diff = abs(max_val - values[p])
            if diff > theta:
                prioritySweep.update(p, -diff)

    return values

#            