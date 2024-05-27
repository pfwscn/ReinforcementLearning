#######################################################################
# Copyright (C)                                                       #
# 2024 SK (github.com/pfwscn)                                         #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

#######################################################################
# Value iteration (VI)
# Combines in each sweep (one update of each state), one PE sweep and one PI sweep,
#   by updating state-value (s) with value of the best action q(s, a*);
#   since PI will update policy pi(a*|s), and
#   next PE sweep will start with v(s) = q(s, a*).
# Instead of expectation over all actions, use max (Bellman optimality)
# v_k+1(s) = max_a sum over s', r p(s', r|s, a) [r + gamma v_k(s') (4.10)
# "value iteration is obtained simply by turning the Bellman optimality
# equation into an update rule ... value iteration update is identical
# to the policy evaluation update (4.5) except that it requires the
# maximum to be taken over all actions... stop once the value function
# changes by only a small amount in a sweep."
# Faster convergence often achieved by interposing multiply PE sweeps
# between each PI sweep. (? gambler's problem pe=1 fastest)
#
# GPI General Policy Iteration
# "general idea of letting policy-evaluation and policy-improvement
# processes interact, independent of their details"
# "the value function stabilizes only when it is consistent with the current policy,
# and the policy stabilizes only when it is greedy with respect to the current value function"
#
# "On problems with large state spaces, asynchronous DP methods are often preferred
# ... the problem is still potentially solvable because relatively few states occur along
# optimal solution trajectories"
#
# bootstrapping: "update estimates on the basis of other estimates"
#
#######################################################################
# Example 4.3 Undiscounted, episodic, finite MDP
# Make bets on the outcomes of a sequence of coin flips with initial capital of c dollars
# Terminal states: reach goal of c = 100 dollars, or lose all money, c = 0
# Non-terminal states: s in {1, 2, ... 99}, tracks amount of capital c 
# Actions available at state s: 
#   stake a in {0, 1, ..., min(s, 100-s) } dollar amount that kth flip is Heads
# Since goal is fixed at 100, 'over-staking' a bet is unnecessary risk, 
#   e.g. in s=70, only necessary to stake at most 30 (and avoid losing 70!)
# Next state: receives (s+x) amount of dollars (winnings) if kth flip is Heads, 
#   otherwise lose stake (s-x)
# Reward: +1 when goal c=100 is reached, and 0 otherwise
# Value iteration (4.10)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('Agg')

GOAL = 100
# all states, including terminal states 0 and 100; there are 101 states
STATES = list(range(GOAL+1))
HEAD_PROB = 0.45   # probability of heads (win)

def figure_4_3():
    # state value
    state_value = np.zeros(GOAL + 1)
    state_value[GOAL] = 1.0
    sweeps_history = []
    # value iteration
    while True:
        old_state_value = state_value.copy()
        sweeps_history.append(old_state_value)
        for state in STATES[1:GOAL]:
            # get possilbe actions for current state
            actions = np.arange(min(state, GOAL - state) + 1)
            action_returns = []
            for action in actions:
                action_returns.append(
                    HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action])
            new_value = np.max(action_returns)
            state_value[state] = new_value
        delta = abs(state_value - old_state_value).max()
        if delta < 1e-9:
            sweeps_history.append(state_value)
            break
    # compute the optimal policy
    policy = np.zeros(GOAL + 1)
    for state in STATES[1:GOAL]:
        actions = np.arange(min(state, GOAL - state) + 1)
        action_returns = []
        for action in actions:
            action_returns.append(
                HEAD_PROB * state_value[state + action] + (1 - HEAD_PROB) * state_value[state - action])
        # round to resemble the figure in the book, see
        # https://github.com/ShangtongZhang/reinforcement-learning-an-introduction/issues/83
        policy[state] = actions[np.argmax(np.round(action_returns[1:], 5)) + 1]

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for sweep, state_value in enumerate(sweeps_history):
        plt.plot(state_value, label='sweep {}'.format(sweep))
    plt.xlabel('Capital')
    plt.ylabel('Value estimates')
    plt.legend(loc='best')
    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel('Capital')
    plt.ylabel('Final policy (stake) ' + str(HEAD_PROB))
    plt.savefig('images/figure_4_3_' + str(HEAD_PROB) + '.png')
    plt.close()

#######################################################################
ACTIONS = []  # ignoring stake 0 since nothing happens
# number of actions increases then decreases symmetrically, peaks at s=50
# number of actions determine interactions between states, info. flow between states
# todo: action_values
def setup_actions():
    ACTIONS.append([])  # terminal state 0
    for s in STATES[1:GOAL]:  # non-terminal states 1...99        
        actions = range(1, min(s, GOAL-s) + 1)  # sensible/possible action range for a state
        # print (s, len(actions), actions)
        ACTIONS.append(actions)
    ACTIONS.append([])  # terminal state 100    

rng = np.random.default_rng()
# argmax with ties broken arbitrarily
def multi_argmax(values):
    # print(values)
    max_val = np.max(values)    
    max_args = np.where(np.abs(values - max_val) < 1e-9)[0].tolist()  # comparing reals    
    max_arg = rng.choice(max_args)
    return max_arg, max_val, max_args

# ph probability of heads; affects concavity of estimated state-value curve
# state-value function gives prob of winning from each state; higher prob, closer to goal state    
# pe number of PE sweeps before a PI; pe=1 value iteration
# pi type of policy: book (lowest best), single (random best)
# value iteration, inplace update
def my_figure_4_3(ph, pi="book"):
    if len(ACTIONS) == 0: setup_actions()
    state_values = np.zeros(GOAL + 1)      
    state_values[GOAL] = 1.0  # if (s+a) == GOAL: g += ph # reward for reaching goal terminal state
    sweeps = []
    sweeps.append(state_values.copy())
    k = 0   # value iterations v_k+1(s) = max_a q(s, a), combines PE + PI in one step
    while True:
        delta = 0
        for s in STATES[1:GOAL]:  # non-terminal states 1...99
            actions = list(ACTIONS[s])
            returns = []    # expected return for taking action a from state s
            for a in actions:   # 0 reward to collect, next state is either s+a or s-a
                g = ph * state_values[s+a] + (1.0-ph) * state_values[s-a]                
                returns.append(g)            
            new_value = np.max(returns)
            delta = np.max([delta, abs(new_value - state_values[s])])
            state_values[s] = new_value            
        # finished a sweep, all state values are updated
        sweeps.append(state_values.copy())
        k += 1
        if delta < 1e-9:            
            # print(49, state_values[49])
            # print(50, state_values[50])
            # print(51, state_values[51])
            break
    print("my Fig. 4.3 " + str(ph) + " " + str(k))
    
    # policy upon state-value convergence
    # policy is a mapping from levels of capital to stakes
    # the optimal policy maximizes the prob. of reaching the goal
    policy = np.zeros(GOAL + 1)
    for s in STATES[1:GOAL]:  # reversed
        actions = list(ACTIONS[s])  # np.arange(min(s, GOAL-s) + 1)
        returns = []
        for a in actions:
            returns.append(ph * state_values[s+a] + (1.0-ph) * state_values[s-a]) 
        if "book" == pi:
            policy[s] = actions[np.argmax(np.round(returns, 5))] # best action with lowest index
        elif "single" == pi:
            max_arg = multi_argmax(np.round(returns, 5))[0]  
            policy[s] = actions[max_arg]

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for i, sweep in enumerate(sweeps):  # reversed
        if i == k or i < 10: 
            plt.plot(sweep, label='sweep {}'.format(i))
    plt.xlabel('Capital')
    plt.ylabel('State-value estimates')
    plt.grid(True)
    plt.legend(loc='best')    

    plt.subplot(2, 1, 2)
    plt.scatter(STATES, policy)
    plt.xlabel('Capital')    
    plt.ylabel('Final policy (stake) ' + str(ph) + ' ' + pi)
    plt.grid(True)
    plt.savefig('images/my_figure_4_3_' + str(ph) + '_' + pi + '.png')    
    plt.close()


# pe number of PE sweeps (per state) before a PI; pe=1 value iteration
# pi type of policy: all (all best)
def many_figure_4_3(ph, pe=1, pi="all"):
    if len(ACTIONS) == 0: setup_actions()
    state_values = np.zeros(GOAL + 1)      
    state_values[GOAL] = 1.0
    update_counts = np.zeros(GOAL + 1)
    sweeps = []
    sweeps.append(state_values.copy())
    k = 0
    while True:
        delta = 0        
        for s in STATES[1:GOAL]:
            actions = list(ACTIONS[s])
            returns = [] 
            for a in actions:                
                returns.append(ph * state_values[s+a] + (1.0-ph) * state_values[s-a])
            # sweep completed for this state, increment count here to be consistent with value iteration 
            update_counts[s] += 1            
            if update_counts[s] >= pe:            
                new_value = np.max(returns)
                update_counts[s] = 0
            else:
                new_value = np.mean(returns)  # expected return, all actions equi-probable
            # new_value = np.mean(returns)  # pure PE
            delta = np.max([delta, abs(new_value - state_values[s])])
            # stick to the best policy so far, for convergence when pe > 1
            state_values[s] = max(state_values[s], new_value)  
            # state_values[s] = new_value # pure PE
        # finished a sweep over all states, all state values are updated
        sweeps.append(state_values.copy())
        k += 1
        # print(k, delta)
        if delta < 1e-9:            
            break
    print("many Fig. 4.3 {}".format(k))
    
    policy = []
    policy.append([0])  # state 0
    for s in STATES[1:GOAL]:  # non-terminal states 1...99
        actions = np.array(list(ACTIONS[s]))  # take advantage of "fancy" indexing
        returns = []
        for a in actions:
            returns.append(ph * state_values[s+a] + (1.0-ph) * state_values[s-a])
        _, _, max_args = multi_argmax(np.round(returns, 5))
        best_actions = actions[max_args].tolist()
        policy.append(best_actions)
    policy.append([0])  # state 100
    
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for i, sweep in enumerate(sweeps):
        if i == k or i < 10: 
            plt.plot(sweep, label='sweep {}'.format(i))
    plt.xlabel('Capital')
    plt.ylabel('State-value estimates ph={} pe={}'.format(ph, pe))
    plt.legend(loc='best')
    plt.grid(True, 'both')
    plt.subplot(2, 1, 2)
    
    x = np.concatenate([np.repeat(s, len(p)) for s, p in enumerate(policy)])
    y = np.concatenate(policy)
    # print(x.size, y.size); print(x); print(y)
    plt.scatter(x, y)     
    plt.xlabel('Capital')    
    plt.ylabel('Final policy (stake) ph={} pe={}'.format(ph, pe))
    plt.grid(True)
    plt.savefig('images/many_figure_4_3_' + str(ph) + '_' + str(pe) + '.png')  
    plt.close()

#######################################################################

if __name__ == '__main__':
    figure_4_3()
    
    # test multi_argmax
    # v = [0.02, 0.02223, 0.02, 0.0199]
    # print(multi_argmax(np.round(v, 1)))
    # print(multi_argmax(np.round(v, 1)))
    # print(multi_argmax(np.round(v, 2)))
    # print(multi_argmax(np.round(v, 2)))    
    # print(multi_argmax(np.round(v, 3)))
    # print(multi_argmax(np.round(v, 3)))

    # setup_actions()
    # estimate value=ph at s=50, in sweep#=pe (when max first applied)
    my_figure_4_3(0.4) # k=17
    my_figure_4_3(0.4, pi="single") # k=17
    
    # should all produce identical converged estimated values curve and policy
    many_figure_4_3(0.4, pe=1) # k=17
    many_figure_4_3(0.4, pe=2) # k=32
    many_figure_4_3(0.4, pe=3) # k=48
    
    phs = [0.25, 0.55, 0.75]  # k=14, 1628, 118
    for p in phs:        
        many_figure_4_3(p)

#######################################################################    