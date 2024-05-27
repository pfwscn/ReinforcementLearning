#######################################################################
# Copyright (C)                                                       #
# 2024 SK (github.com/pfwscn)                                         #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

#######################################################################
# Chp 4 Dynamic Programming (DP)
# DP algos turn Bellman equations into update rules
# Compute value functions using iterative solution methods
#
# PE Policy Evaluation (Prediction)
# "Purpose is to compute value functions for a policy to help find better policies."
# (in-place) iterative policy evaluations
#   convergence of the sequence of values {v_k} to v_pi as k->inf
# expected update: state(action)-value updates are based on 
#   "an expectation over all possible next states rather than 
#   on a sample next state"
#
# PI Policy Improvement 
# "the process of making a new policy that improves on an original policy,
# by making it greedy with respect tot eh value function of the original policy"
# PI theorem : q_pi(s, newpi(s)) >= v_pi(s) (4.7)
# v_newpi(s) >= v_pi(s) for all s (4.8)
# greedy policy: take the action that looks best at each state (4.9)
#   ties broken arbitrarily
# "PI must give us a strictly better policy except when the original policy
# is already optimal."
#
# Policy iteration
# A way of finding an optimal policy:
#   successive application of PE + PI to produce a sequence of
#   monotonically improving policies and value functions
# In policy iteration, each policy evaluation, itself an iterative computation,
#   is started with the value function for the previous policy.
# "Because a finite MDP has only a finite number of deterministic policies,
# this process must converge to an optimal policy and the optimal value function
# in a finite number of iterations."
#
# Drawback of explicit separate PE and PI phases, 
#   wait until PE convergence (of state-values, v_pi)
#   Example 4.1 suggests PE phase maybe truncated without losing
#   convergence guarantees of policy iteration.
#
#######################################################################
# Example 4.1

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table
matplotlib.use('Agg')

# top-left is [0,0]
WORLD_SIZE = 4
# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTIONS_FIGS=[ '←', '↑', '→', '↓']
ACTION_PROB = 0.25

# the terminal state covers top-left and bottom-right cells
def is_terminal(state):
    x, y = state
    return (x == 0 and y == 0) or (x == WORLD_SIZE - 1 and y == WORLD_SIZE - 1)

# the reward is -1 for all actions (on all transitions), except in the terminal state
def step(state, action):
    if is_terminal(state):
        return state, 0    
    next_state = (state + action).tolist()
    x, y = next_state    
    # gone off-grid
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        next_state = state
    reward = -1
    return next_state, reward

# Policy evaluation
# Episodic task, discount (gamma) can be 1.0 (3.11)
def evaluate_policy(initial_values, in_place = True, discount = 1.0, 
                    start_step=0, max_steps=1000):
    state_values = initial_values.copy()
    if not in_place:
        new_state_values = np.zeros_like(state_values)
    k = start_step
    while True:
        # initialize delta here 'cos interested in max delta for each sweep of states, 
        # not over all sweeps
        delta = 0  
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                value = 0
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)
                    value += ACTION_PROB * (reward + discount * state_values[next_i, next_j])
                delta = np.max([delta, abs(state_values[i, j] - value)])
                if not in_place:
                    new_state_values[i, j] = value
                else:
                    state_values[i, j] = value
        if not in_place:  # update state values with copy of new values
            state_values = new_state_values.copy()  
        k += 1  # finished an interation, so update k here
        # compare max difference in state values this sweep made, continue policy evaluation?
        if delta < 1e-4 or k >= max_steps:  
            break
    return state_values, k

# def compute_state_value(in_place = True, discount = 1.0):
#     new_state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))    
#     iteration = 0
#     while True:
#         if in_place:
#             state_values = new_state_values
#         else:
#             state_values = new_state_values.copy()
#         old_state_values = state_values.copy()  # me: for terminating condition, delta
#         for i in range(WORLD_SIZE):
#             for j in range(WORLD_SIZE):
#                 value = 0
#                 for action in ACTIONS:
#                     (next_i, next_j), reward = step([i, j], action)
#                     value += ACTION_PROB * (reward + discount * state_values[next_i, next_j])
#                 new_state_values[i, j] = value
#         max_delta_value = abs(old_state_values - new_state_values).max()
#         if max_delta_value < 1e-4:
#             break
#         iteration += 1
#     return new_state_values, iteration


# sync and async (in-place) state-values differ at the beginning (small k), but
#   converge to the same v_pi(s) values with larger k
#
# Section 4.5 Asynchronous (sweepless) DPs are iterative DPs algos, that update state values
# not in systematic sweeps, but in any order and relative counts (some states may be
# updated multiple times before others are updated once), using whatever values available.
# However, for convergece, all states need to be updated continuously, 
# i.e. no state ignored after a point in time.
# This greater flexibility enables strategized ordering of state-value updates,
# to stimulate more efficient propagation of state-value information, relevant 
# to optimal behavior.
def figure_4_1(): 
    m = [0, 3, 10, 1000]   
    state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))     
    for i, j in list(zip(m[:-1], m[1:])):
        state_values, iterations = evaluate_policy(state_values, in_place = True, 
                                                   start_step=i, max_steps=j)
        draw_image(np.round(state_values, decimals=2)) 
        plt.savefig('images/my_figure_4_1_async_' + str(iterations) + '.png')
        plt.close()
        print('Async: {} iterations'.format(iterations))    
    state_values = np.zeros((WORLD_SIZE, WORLD_SIZE)) 
    for i, j in list(zip(m[:-1], m[1:])):
        state_values, iterations = evaluate_policy(state_values, in_place = False, 
                                                   start_step=i, max_steps=j)
        draw_image(np.round(state_values, decimals=2)) 
        plt.savefig('images/my_figure_4_1_sync_' + str(iterations) + '.png')
        plt.close()
        print('Sync: {} iterations'.format(iterations))
    # async_values, async_iteration = compute_state_value(in_place = True) # 113
    # plt.savefig('images/my_figure_4_1_async0.png') 
    # sync_values, sync_iteration = compute_state_value(in_place = False)  # 172
    # plt.savefig('images/my_figure_4_1_sync0.png')

    
def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows
    # Add cells
    for (i, j), val in np.ndenumerate(image):
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')
        # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')
    ax.add_table(tb)

#######################################################################
# Exercise 4.5 Policy iteration with action-values

# Action-values for terminal states = 0
# Need to compute for non-terminal states only
ST = [ (0, 0), (WORLD_SIZE-1, WORLD_SIZE-1) ]  # terminal states
NUM_ACTIONS = len(ACTIONS)
PI = 0.25
# in-place aciton-value updates
# p(s',r| s, a) = 1.0
# q_pi(s, a) = sum over all s' sum over all r p(s',r| s, a) [r + gamma * sum over all a' in s' pi(a'|s') q_pi(s', a')]
def policy_evaluation(initial_values, start_step=0, max_steps=1000, gamma = 1.0):
    action_values = initial_values.copy()
    k = start_step
    while True:
        delta = 0  
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                if (i, j) in ST:
                    continue                
                for a, action in enumerate(ACTIONS):
                    (next_i, next_j), reward = step([i, j], action)
                    q_value = reward + gamma * np.sum(PI * action_values[next_i, next_j])                    
                    delta = np.max([delta, abs(action_values[i, j, a] - q_value)])
                    action_values[i, j, a] = q_value
        k += 1        
        if delta < 1e-4 or k >= max_steps:  
            break
    return action_values, k
     
# Find the step when a stable (optimal) policy is first found
# Greedy PI follows after one sweep of PE
def policy_iteration():
    action_values = np.zeros((WORLD_SIZE, WORLD_SIZE, NUM_ACTIONS))
    action_values, iterations = policy_evaluation(action_values, start_step=0, max_steps=1)
    best_values, old_policy = get_best(action_values)
    for i in range(1, 1000):
        action_values, iterations = policy_evaluation(action_values, start_step=i, max_steps=i+1)        
        best_values, best_policy = get_best(np.round(action_values, decimals=2))
        if np.all(old_policy == best_policy):
            draw_image(np.round(best_values, decimals=2)) 
            plt.savefig('images/my_PI_q_' + str(iterations) + '.png')
            plt.close()
            draw_policy_q(best_policy)
            plt.savefig('images/my_PI_q_policy_' + str(iterations) + '.png')
            plt.close()                    
            print(iterations)
            return
        else:
            old_policy = best_policy.copy()         
    print("Not found")

# chooses the best action(s) in each state
def get_best(qvalues):
    best_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    best_policy = np.full((WORLD_SIZE, WORLD_SIZE), '', dtype=object)
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):
            if (i, j) in ST:
                continue
            best_val = np.max(qvalues[i, j])
            best_values[i, j] = best_val                        
            for a in np.where(best_val == qvalues[i, j])[0]:
                best_policy[i, j] += ACTIONS_FIGS[a]            
            # print(i, j, qvalues[i, j], best_val, best_policy[i, j])            
    return best_values, best_policy

def draw_policy_q(best_policy):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    nrows, ncols = best_policy.shape
    width, height = 1.0 / ncols, 1.0 / nrows
    # Add cells
    for (i, j), val in np.ndenumerate(best_policy):
        tb.add_cell(i, j, width, height, text=val,
                loc='center', facecolor='white')
    # Row and column labels...
    for i in range(len(best_policy)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                   edgecolor='none', facecolor='none')
    ax.add_table(tb)


def figure_4_1q(): 
    m = [0, 3, 4, 5, 6, 7, 8, 9, 10, 1000]   
    action_values = np.zeros((WORLD_SIZE, WORLD_SIZE, NUM_ACTIONS))
    for i, j in list(zip(m[:-1], m[1:])):
        action_values, iterations = policy_evaluation(action_values, start_step=i, max_steps=j)
        best_values, best_policy = get_best(np.round(action_values, decimals=2))            
        draw_image(np.round(best_values, decimals=2)) 
        plt.savefig('images/my_figure_4_1q_' + str(iterations) + '.png')
        plt.close()
        draw_policy_q(best_policy)
        plt.savefig('images/my_figure_4_1q_policy_' + str(iterations) + '.png')
        plt.close()                    
        print(iterations)        

#######################################################################

if __name__ == '__main__':
    figure_4_1()
    policy_iteration()
    figure_4_1q()    

#######################################################################