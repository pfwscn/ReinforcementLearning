#######################################################################
# Copyright (C)                                                       #
# 2024 SK (github.com/pfwscn)                                         #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table
matplotlib.use('Agg')

#######################################################################
# Chp 3 Finite MDPs (Markov Decision Processes)
#
# RL: Learning from experience 
#   (interaction with an environment, and receiving evaluative feedback on action taken) 
# to associate best actions with situations (policy) 
# to achieve a goal (expected cumulative reward over the long term).
#
# MDP - a perfect model of a RL environment.
# "MDPs are a classical formalization of sequential decision making,
# where actions influence not just immediate rewards, but also 
# subsequent situations or states, and through those, future rewards."
# Trade-off between immediate/present and delayed/future rewards.
# Evaluate state-dependent long-term consequences (expected future rewards) 
# of taking actions
# Value of an action depends on state q(s,a) 
# Value of a state depends on best action from state v(s) = q(s, a*)
#
# A finite MDP problem is defined by finite sets (States, Actions, Rewards) 
# that over discrete timesteps t=0,1,... interact to produce trajectories 
# S_0, A_0, R_1, S_1, A_1, R_2, S_2, ..., S_n, A_n, R_n+1, ... S_T (3.1) 
# MDP dynamics is governed by the Markov property of random variables 
# S_t+1 and R_t+1, that depend on preceeding state and action S_t, A_t
# only (3.2) p(s', r | s, a)
#
# Example 3.3 Transition graph: state and action nodes
# Multiple actions are possible from a state.
# The outcome (s', r) of an action taken from a state (s, a) is 
# probabilitistic (3.2); there can have more than one transition 
# from a (s, a) each ending in a different (s', r),
# but their probabilities must sum to 1.0 (3.3). 
#
# (3.4) state transition probability p(s' | s, a)
# (3.5) expected *reward* for a state-action pair r(s, a) //total expectation
# (3.6) expected reward for state-action-nextstate triple r(s, a, s')
#
# "The reward signal is your way of communicating to the agent 
# _what_ your want to achieved, not _how_ you want it achieved." 
# Section 17.4 designing effective reward signals
#
# Expected *return* G_t, is a function of the reward sequence 
# after time t, R_t+1, R_t+2,...R_T (3.7) // future expected return
#
# T is the final timestep. Two types of tasks: episodic, continuing.
# Episodic tasks end in a special state (the terminal state);
# episodes are independent of each other. Continuing tasks, T = inf 
# The terminal state is an absorbing state (transitions only to itself), 
# with zero rewards. (3.11)
#
# Rewards are discounted; discount rate gamma [0.0, 1.0]:
#   0.0, myopic agent (consider present rewards only), 
#   weight recent rewards more heavily than future rewards, 
#   1.0, far-sighted (weight all rewards equally).
#
# Choose action A_t (from S_t) to maximize expected discounted return G_t
# (3.8) G_t = R_t+1 + gamma R_t+2 + gamma^2 R_t+3 + ...
# (3.9) G_t = R_t+1 + gamma G_t+1, with G_T=0
# (3.10) G_t is finite, gamma < 1
#  
# Value function estimates how "good", in terms of expected return,
# it is to be in a certain state or to take a certain action from a state.
# The value of a state is the reward collected by an action, 
# plus the expected value of the next state.
# *Value functions are defined w.r.t policies (pi)* //plan of behavior
# // Value functions "evaluate" policies; they define a partial ordering over policies.
#
# A policy is a mapping from states to probabilities of selecting each possible action.
# pi_t(a|s) = probability of taking action a from state s at time t
#
# State-value function (for pi) v_pi(s) = E_pi[G_t|S_t=s] (3.12)
# The value of a state s, under a policy pi, is the expected return , 
# when starting in s, and following pi thereafter. 
# Recall: MDP sequence: S_0, A_0, R_1,...S_t, A_t, R_t+1, S_t+1,...,S_T (G_T=0)
# reward is the evidence of an action taken (an event); (3.9) G_t = R_t+1 + gamma * G_t+1 
#
# Action-value function (for pi) q_pi(s, a) = E_pi[G_t|S_t=s, A_t=a] (3.13)
# The value of taking action a from state s under policy pi, is the expected return
# starting from s, taking action a, and following pi thereafter.
# Compare backup diagrams for v_pi(s) with q_pi(s, a).
# Value of the (absorbing) terminal state, if any, is always 0; v_pi(S_T)=0 since G_T=0.
#
# The Bellman equation (3.14) expresses the relationship between the value of a state, 
# and the value of its successor states. It must hold for all states (Exercise 3.14).
# The expected version is a weighted average over all possibilities (a, r, s') from s.
# It is used as an update rule (4.5) in iterative policy evaluation.
#
# "Reinforcement learning methods specifies how the agent's *policy* 
# is changed as a result of its experience." //search for an optimal policy
#  
# Policy A is at least as good or better (>=) than policy B if 
#   *for all* states s, v_A(s) >= v_B(s).
# An optimal policy (pi*) always exists.
# Value functions for an optimal policy are optimal in the sense that 
# v*(s) = max over all pi v_pi(s) for all states (3.15) optimal state-value function
# q*(s, a) = max over all pi q_pi(s, a) for all states and actions (3.16) 
#   optimal action-value function (3.17)
#
# The Bellman optimality equation (3.18, 3.19, 3.20, 4.1, 4.2) expresses the fact that 
# the value of a state under an optimal policy, must equal the expected return 
# for the best action from that state (Fig. 3.4) 
# v*(s) = max-of over all actions for s q_pi*(s, a)
# Backup diagram Fig. 3.4
#
#######################################################################

# Example 3.5
# States of the environment: cells of the grid
# Actions possible from each state: north, south, east, west (up, down, right, left)
# Action policy at each state: random with equi-probability pi(a|s) = 0.25 
# State transitions are deterministic: p(s' | s, a) = 1.0
# Rewards: -1 for off-grid actions, 0 for others except from states A and B
# Any action from A transitions to A' with +10 reward p(A', 10 | A, *) 
# Any action from B transitions to B' with +5 reward p(B', 5 | B, *) 

# top-left is [0, 0]
WORLD_SIZE = 5
A_POS = [0, 1]
A_PRIME_POS = [4, 1]
B_POS = [0, 3]
B_PRIME_POS = [2, 3]

# left, up, right, down
ACTIONS = [np.array([0, -1]),
           np.array([-1, 0]),
           np.array([0, 1]),
           np.array([1, 0])]
ACTIONS_FIGS=[ '←', '↑', '→', '↓']
PI = 0.25  # pi(a|s), probability of an action from a state 
GAMMA = 0.9  # discount rate

# (s', r | s, a)
def step(state, action):
    if state == A_POS:
        return A_PRIME_POS, 10
    if state == B_POS:
        return B_PRIME_POS, 5
    next_state = (state + action).tolist()
    x, y = next_state
    if x < 0 or x >= WORLD_SIZE or y < 0 or y >= WORLD_SIZE:
        reward = -1.0
        next_state = state
    else:
        reward = 0
    return next_state, reward

# Expected update: new state-value is expectated return over all actions
def figure_3_2():
    state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))    
    k=0
    while True: # keep iterating until convergence
        next_values = np.zeros_like(state_values)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                # bellman equation
                for action in ACTIONS:     
                    (next_i, next_j), reward = step([i, j], action)                    
                    next_values[i, j] += PI * (reward + GAMMA * state_values[next_i, next_j])
        k += 1
        if np.sum(np.abs(next_values - state_values)) < 1e-4:            
            draw_image(np.round(next_values, decimals=2))
            plt.savefig('images/my_figure_3_2.png')
            plt.close()
            print("Fig 3.2 {}".format(k)) # 77
            break
        state_values = next_values

# new state-value is max return possible over all actions from a state
def figure_3_5():
    state_values = np.zeros((WORLD_SIZE, WORLD_SIZE))    
    k = 0
    while True:
        next_values = np.zeros_like(state_values)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                values = []
                for action in ACTIONS:
                    (next_i, next_j), reward = step([i, j], action)                    
                    # value iteration
                    values.append(reward + GAMMA * state_values[next_i, next_j])
                next_values[i, j] = np.max(values)
        k += 1        
        if np.sum(np.abs(next_values - state_values)) < 1e-4:
            draw_image(np.round(next_values, decimals=2))
            plt.savefig('images/my_figure_3_5.png')
            plt.close()
            draw_policy(next_values)
            plt.savefig('images/my_figure_3_5_policy.png')
            plt.close()
            print("Fig 3.5 {}".format(k)) # 124
            break        
        state_values = next_values


def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows
    # Add cells
    for (i, j), val in np.ndenumerate(image):
        # add state labels
        if [i, j] == A_POS:
            val = str(val) + " (A)"
        if [i, j] == A_PRIME_POS:
            val = str(val) + " (A')"
        if [i, j] == B_POS:
            val = str(val) + " (B)"
        if [i, j] == B_PRIME_POS:
            val = str(val) + " (B')"        
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')        
    # Row and column labels...
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')
    ax.add_table(tb)


# What's the best action(s) to take in a cell given optimal state-values?
# From each cell/state, examine optimal value of all possible successor states.
def draw_policy(optimal_values):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    nrows, ncols = optimal_values.shape
    width, height = 1.0 / ncols, 1.0 / nrows
    # Add cells
    for (i, j), val in np.ndenumerate(optimal_values):
        next_vals=[]
        for action in ACTIONS:
            next_state, _ = step([i, j], action)
            next_vals.append(optimal_values[next_state[0],next_state[1]])
        best_actions=np.where(next_vals == np.max(next_vals))[0]
        val=''
        for ba in best_actions:
            val+=ACTIONS_FIGS[ba]        
        # add state labels
        if [i, j] == A_POS:
            val = str(val) + " (A)"
        if [i, j] == A_PRIME_POS:
            val = str(val) + " (A')"
        if [i, j] == B_POS:
            val = str(val) + " (B)"
        if [i, j] == B_PRIME_POS:
            val = str(val) + " (B')"        
        tb.add_cell(i, j, width, height, text=val,
                loc='center', facecolor='white')
    # Row and column labels...
    for i in range(len(optimal_values)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                   edgecolor='none', facecolor='none')
    ax.add_table(tb)


#######################################################################    
# Use action-value functions q(s, a), instead of state-value functions v(s)
# The result that matters is the policy (what action to take from a state), not the state values
# With q(s, a), policy determined from computed q(s, a), not making another sweep over best state values
# The decision is local/confined to a state at any point in time.
# Fig. 3.4 backup diagram!
# Advantage of q(s, a) over v(s) for action selection (policy improvement) step:
# " At the cost of representing a function of state-action pairs instead of just of states, 
# the optimal action-value function (q*) allows optimal actions to be selected without 
# having to know anything about the environment's dynamics (successor states and their values)."

NUM_ACTIONS = len(ACTIONS)
# Iterative policy evaluation
# Expected update: new action-value is reward plus (discounted) expectated return over all successor actions
# over all actions from the next state(s) (4.6)
def figure_3_2q():     
    action_values = np.zeros((WORLD_SIZE, WORLD_SIZE, NUM_ACTIONS))    
    k = 0
    while True:
        next_qvalues = np.zeros_like(action_values)               
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                for a, action in enumerate(ACTIONS):
                    (next_i, next_j), reward = step([i, j], action)
                    # q(s, a) = reward + discounted value of s'
                    # value of s' is expected return over all possible actions from s'
                    next_qvalues[i, j, a] = reward + GAMMA * np.sum(PI * action_values[next_i, next_j])
        k += 1        
        if np.sum(np.abs(next_qvalues - action_values)) < 1e-4:
            optimal_values, optimal_policy = get_best(np.round(next_qvalues, decimals=2))            
            draw_image(optimal_values)
            plt.savefig('images/my_figure_3_2q.png')
            plt.close()            
            draw_policy_q(optimal_policy)
            plt.savefig('images/my_figure_3_2q_policy.png')
            plt.close()
            print("Fig 3.2q {}".format(k)) # 90
            break        
        action_values = next_qvalues

# update: new action-value is reward plus max return possible over all successor actions
# over all actions from the next state (3.20)
# p(s', r | s, a) = 1.0
# same values as fig 3.5; v*(s) = max over a q_pi*(s, a)        
def figure_3_5q():
    action_values = np.zeros((WORLD_SIZE, WORLD_SIZE, NUM_ACTIONS))    
    k = 0
    while True:
        next_qvalues = np.zeros_like(action_values)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):                
                for a, action in enumerate(ACTIONS):
                    (next_i, next_j), reward = step([i, j], action)
                    # qvalue iteration
                    next_qvalues[i, j, a] = reward + GAMMA * np.max(action_values[next_i, next_j])
        k += 1        
        if np.sum(np.abs(next_qvalues - action_values)) < 1e-4:
            optimal_values, optimal_policy = get_best(np.round(next_qvalues, decimals=2))            
            draw_image(optimal_values)
            plt.savefig('images/my_figure_3_5q.png')
            plt.close()            
            draw_policy_q(optimal_policy)
            plt.savefig('images/my_figure_3_5q_policy.png')
            plt.close()            
            print("Fig 3.5q {}".format(k)) # 139 takes longer to converge, more values involved
            break        
        action_values = next_qvalues


# Advantage of q(s, a) over v(s)
# To retrieve best policy, only need to inspect qvalues, 
# no call to step() to get the next state, and its value.
def get_best(qvalues):
    best_values = np.zeros((WORLD_SIZE, WORLD_SIZE))
    best_policy = np.full((WORLD_SIZE, WORLD_SIZE), '', dtype=object)
    for i in range(WORLD_SIZE):
        for j in range(WORLD_SIZE):                
            best_val = np.max(qvalues[i, j])
            best_values[i, j] = best_val                        
            for a in np.where(best_val == qvalues[i, j])[0]:
                best_policy[i, j] += ACTIONS_FIGS[a]            
            # print(i, j, qvalues[i, j], best_val, best_policy[i, j])            
    return best_values, best_policy

def draw_policy_q(optimal_policy):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])
    nrows, ncols = optimal_policy.shape
    width, height = 1.0 / ncols, 1.0 / nrows
    # Add cells
    for (i, j), val in np.ndenumerate(optimal_policy):        
        # add state labels
        if [i, j] == A_POS:
            val = str(val) + " (A)"
        if [i, j] == A_PRIME_POS:
            val = str(val) + " (A')"
        if [i, j] == B_POS:
            val = str(val) + " (B)"
        if [i, j] == B_PRIME_POS:
            val = str(val) + " (B')"        
        tb.add_cell(i, j, width, height, text=val,
                loc='center', facecolor='white')
    # Row and column labels...
    for i in range(len(optimal_policy)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                   edgecolor='none', facecolor='none')
    ax.add_table(tb)


if __name__ == '__main__':    
    figure_3_2()
    figure_3_5()    
    figure_3_2q()
    figure_3_5q()

#######################################################################