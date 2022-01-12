import random
from statistics import mean


#--------------------------------------- Minimax ----------------------------------------#
def minimax(gameState, depth, utility):
    """ The minimax algorithm searches for the optimal move to play for the given game
    state. Multiple players may be playing the game taking turns one at a time. For every
    max move we have multiple min moves corresponding to the opponent moves. The algorithm
    assumes that the opponents will also play optimally.
    The minimax algorithm is a search algorithm trying all actions alternating between
    players. The algorithm deepens recursively until it reaches a terminal node and then
    backs up the value. 
    The game tree is expanded to an arbitrary depth. A ply is single move by *one* of the
    players. Game search of depth 1 is considered to involve every player making a ply.
    When reaching the limiting depth of the search tree, the leaves are scored with the
    provided utility funcion.
    Returns the action maximizing the score of the player under adversarial play of the
    other players.

    @param depth (int): Maximum depth of the search tree.
    @param utility (func): An evaluation function used to score the leaves of the search
            tree in case they are not terminal states.
    @return action (Any): The action that is the optimal choice for the player.
    """
    # Collect legal moves and successor states scores.
    legalMoves = gameState.actions()
    scores = [_minimax(gameState.result(act), 1, depth, utility) for act in legalMoves]
    # Choose one of the best actions.
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return legalMoves[chosenIndex]

def _minimax(gameState, current_depth, depth, utility):
    # Check if the recursion has reached the bottom.
    if gameState.is_terminal():
        return gameState.score()
    if gameState.agent_id() == gameState.to_move() and depth <= current_depth:
        return utility(gameState)
    # Generate successor states and evaluate each successor.
    legalMoves = gameState.actions()
    successors = (gameState.result(act) for act in legalMoves)
    if gameState.agent_id() == gameState.to_move():
        return max((_minimax(s, current_depth+1, depth, utility) for s in successors))
    return min((_minimax(s, current_depth, depth, utility) for s in successors))


#-------------------------------------- Alpha-beta --------------------------------------#
""" The effectiveness of alpha-beta pruning is highly dependent on the order in which the
states are examined. This suggests that it might be worthwhile to try to first examine the
successors that are likely to be best. If this could be done perfectly, then alpha-beta
would need to examine only O(b^(m/2)) nodes instead of O(b^m). With random move ordering,
the total number of nodes examined will be roughly O(b^(3m/4)).
A good move-ordering scheme is trying first the moves that were found to be best in the
past. The past could come from previous exploration of the current move through a process
of iterative deepening.
Redundant paths to repeated states can cause an exponential increase in search cost, and
keeping a table of previously reached states can address this problem.
Even with alpha-beta pruning and clever move ordering, minimax won't work for games like
chess and go, because there are still too many states to explore. In the very first paper
on computer game-playing, **Programming a Computer for Playing Chess** (1950), Claude
Shannon recognized this problem and proposed two stratedies:
 - Type A: Consider all possible moves to a certain depth in the search tree, and then use
   a heuristic evaluation function to estimate the utility of states at that depth.
 - Type B: Ignore moves that look bad, and follow promising lines 'as far as possible'. It
   explores a deep but narrow portion of the tree.
Alpha-beta pruning prunes branches of the tree that can have no effect on the final
evaluation, while forward pruning prunes moves that appear to be poor moves. In Shannon's
terms, this is a Type B strategy. One approach to forward pruning is beam search: on each
ply, consider only a 'beam' of the n best moves according to the evaluation function.
"""

def alpha_beta(gameState, depth, utility):
    """ The alpha-beta algorithm is a minimax algorithm using the alpha-beta pruning
    technique.
    Alpha-beta uses two additional parameters:
     - alpha: the value of the best choice we have found so far (maximizing player return)
     - beta: the value of the worst choice we have found so far (maximizing opponent return)
    Alpha-beta search updates the values of alpha and beta as it goes along and prunes the
    remaining branches at a node as soon as the value of the current node is known to be
    worse than the current alpha or beta respectively.
    Returns the action maximizing the score of the player under adversarial play of the
    other players.

    @param depth (int): Maximum depth of the search tree.
    @param utility (func): An evaluation function used to score the leaves of the search
            tree in case they are not terminal states.
    @return action (Any): The action that is the optimal choice for the player.
    """
    # Collect legal moves and successor states scores.
    legalMoves = gameState.actions()
    alpha = -float("inf")
    scores = []
    for act in legalMoves:
        s = gameState.result(act)
        val = _alpha_beta(s, 1, depth, utility, alpha, float("inf"))
        scores.append(val)
        alpha = max(alpha, val)
    # Choose one of the best actions.
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return legalMoves[chosenIndex]

def _alpha_beta(gameState, current_depth, depth, utility, alpha, beta):
    # Check if the recursion has reached the bottom.
    if gameState.is_terminal():
        return gameState.score()
    if gameState.agent_id() == gameState.to_move() and depth <= current_depth:
        return utility(gameState)
    # Generate successor states and evaluate each successor.
    legalMoves = gameState.actions()
    successors = (gameState.result(act) for act in legalMoves)
    if gameState.agent_id() == gameState.to_move():
        return player_eval(successors, current_depth+1, depth, utility, alpha, beta)
    return enemy_eval(successors, current_depth, depth, utility, alpha, beta)

def player_eval(successors, current_depth, depth, utility, alpha, beta):
    max_v = -float("inf")
    for s in successors:
        max_v = max(max_v, _alpha_beta(s, current_depth, depth, utility, alpha, beta))
        if beta < max_v:
            return max_v
        alpha = max(alpha, max_v)
    return max_v

def enemy_eval(successors, current_depth, depth, utility, alpha, beta):
    min_v = float("inf")
    for s in successors:
        min_v = min(min_v, _alpha_beta(s, current_depth, depth, utility, alpha, beta))
        if alpha > min_v:
            return min_v
        beta = min(beta, min_v)
    return min_v


#-------------------------------------- Expectimax --------------------------------------#
def expectimax(gameState, depth, utility):
    """ Expectimax models probabilistic behavior of opponents. While minimax assumes that
    opponents make optimal decisions, expectimax assumes that opponents choose their
    actions uniformly at random from the set of legal actions.

    @param depth (int): Maximum depth of the search tree.
    @param utility (func): An evaluation function used to score the leaves of the search
            tree in case they are not terminal states.
    @return action (Any): The action that is the optimal choice for the player.
    """
    # Collect legal moves and successor states scores.
    legalMoves = gameState.actions()
    scores = [_expectimax(gameState.result(act), 1, depth, utility) for act in legalMoves]
    # Choose one of the best actions.
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best
    return legalMoves[chosenIndex]

def _expectimax(gameState, current_depth, depth, utility):
    # Check if the recursion has reached the bottom.
    if gameState.is_terminal():
        return gameState.score()
    if gameState.agent_id() == gameState.to_move() and depth <= current_depth:
        return utility(gameState)
    # Generate successor states and evaluate each successor.
    legalMoves = gameState.actions()
    successors = (gameState.result(act) for act in legalMoves)
    if gameState.agent_id() == gameState.to_move():
        return max((_expectimax(s, current_depth+1, depth, utility) for s in successors))
    return mean((_expectimax(s, current_depth, depth, utility) for s in successors))

#