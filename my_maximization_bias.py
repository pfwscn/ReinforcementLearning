#######################################################################
# Copyright (C)                                                       #
# 2024 SK (github.com/pfwscn)                                         #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

#######################################################################
# Chp 6 Temporal-Difference (TD) Learning
# Section 6.7 Maximization bias and double learning
# The use of maximization directly (Q-learning), or
#   to select an action from a state (epsilon greedy in Sarsa) to update state-action values
#   can introduce a positive bias into policy evaluation, making finding 
#   the optimal target policy more difficult. 
# Maximization bias occurs when the max of the true values is 0, but max of the estimates is > 0.
#   Using max (greedy policy) of estimates, "hides" the max of true values.
#
# Use double learning to split the identification of the maximizing action, and its update value.
#   Q values are split into two parts: Q1 and Q2. 
#   To update Q1, use Q1 to identify action a from S_t+1 for target policy, 
#   but use Q2 value of a from S_t+1; and vice versa.
# For (double) Q-learning
# Target for Q1(S_t, A_t): R_t+1 + gamma * Q2(S_t+1, argmax Q1(S_t+1, A_t+1 = a)) (6.10)
# Target for Q2(S_t, A_t): R_t+1 + gamma * Q1(S_t+1, argmax Q2(S_t+1, A_t+1 = a))
# To choose next action in an episode (behavior policy), use both Q1 and Q2 values in some fashion
# (e.g. sum or avg); double learning is for value update only (to build/learn the target policy).
#
# Section 6.8 Afterstates and afterstate value functions
# 
#######################################################################

import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng()

# Example 6.7 maximization bias can harm the performance of TD control algos
# MDP: D(3) <= B(1) <- A(0) -> G(2)
# start at A; from A, can go left or right with reward = 0 either way
# but going left from A sets up negative rewards (expected return is negative), 
# while expected return going right from A is 0
# reward for taking left from B is drawn from a distribution with a negative expected value
# Both GOAL and DECOY states are terminal
DECOY_STATE, STATE_B, STATE_A, GOAL_STATE = 3, 1, 0, 2

# ACTIONS
A_LEFT, A_RIGHT = 1, 0
B_LEFTS = range(0, 10)  # multiple actions in B to D
STAY = 0    # no action from decoy or goal states
ACTIONS = [[A_RIGHT, A_LEFT], B_LEFTS, [STAY], [STAY]]

EPSILON = 0.1
# choose an action from a state, with a epsilon greedy algorithm
def choose_action(Q_values, state, epsilon=EPSILON):
    if rng.random() < epsilon:
        action = rng.choice(ACTIONS[state], shuffle=False)
    else:        
        action_values = Q_values[state]
        best_actions = np.where(np.max(action_values) == action_values)[0]
        action = rng.choice(best_actions, shuffle=False)
    return action

# make a step/move from a state with action: S_t, A_t -> R_{t+1}, S_{t+1}
def take_action(state, action):
    if state == STATE_A:
        reward = 0
        next_state = GOAL_STATE if action == A_RIGHT else STATE_B
    elif state == STATE_B:        
        reward = rng.normal(-0.1, 1)
        next_state = DECOY_STATE
    else:# terminal absorbing state
        reward = 0
        next_state = state        
    return reward, next_state


ALPHA = 0.1
GAMMA = 1.0
# Q1_values and Q2_values are lists
# DQL True: use double learning if True
def q_learning(Q1_values, Q2_values=None, DQL=False):
    if DQL: assert Q2_values is not None
    
    aleft_counts = 0  # number of left moves from state A
    state = STATE_A    
    while state not in (GOAL_STATE, DECOY_STATE):
        if DQL:
            Q_values = [q1 + q2 for q1, q2 in zip(Q1_values, Q2_values)]
            action = choose_action(Q_values, state)
        else:
            action = choose_action(Q1_values, state)            
        if state == STATE_A and action == A_LEFT:
            aleft_counts += 1
        reward, next_state = take_action(state, action)
        # S_t, A_t, R_{t+1}, S_{t+1}
        if DQL:
            if rng.random() < 0.5: # (6.10)
                max_action = choose_action(Q1_values, next_state, 0.0)
                target = reward + GAMMA * Q2_values[next_state][max_action]
                Q1_values[state][action] += ALPHA * (target - Q1_values[state][action])                
            else:
                max_action = choose_action(Q2_values, next_state, 0.0)
                target = reward + GAMMA * Q1_values[next_state][max_action]
                Q2_values[state][action] += ALPHA * (target - Q2_values[state][action])            
        else:
            # best action from S_{t+1}
            max_action = choose_action(Q1_values, next_state, epsilon=0.0)
            # Q-Learning update (6.8)
            target = reward + GAMMA * Q1_values[next_state][max_action]            
            Q1_values[state][action] += ALPHA * (target - Q1_values[state][action])
                        
        state = next_state
    return aleft_counts


# Single, Double Sarsa
def sarsa(Q1_values, Q2_values=None, DSL=False):
    if DSL: assert Q2_values is not None
    
    aleft_counts = 0  # number of left moves from state A
    state = STATE_A
    if DSL:
        Q_values = [q1 + q2 for q1, q2 in zip(Q1_values, Q2_values)]
        action = choose_action(Q_values, state)
    else:
        action = choose_action(Q1_values, state)            
    
    while state not in (GOAL_STATE, DECOY_STATE):
        if state == STATE_A and action == A_LEFT:
            aleft_counts += 1        
        reward, next_state = take_action(state, action) 
        # an action from S_{t+1} using Q_t values
        if DSL:
            Q_values = [q1 + q2 for q1, q2 in zip(Q1_values, Q2_values)]
            next_action = choose_action(Q_values, next_state)
        else:
            next_action = choose_action(Q1_values, next_state)        
        # S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
        if DSL:
            if rng.random() < 0.5: # (6.10)  # does not use next_action for update!
                v_action = choose_action(Q1_values, next_state) 
                target = reward + GAMMA * Q2_values[next_state][v_action]
                Q1_values[state][action] += ALPHA * (target - Q1_values[state][action])
            else:
                v_action = choose_action(Q2_values, next_state)
                target = reward + GAMMA * Q1_values[next_state][v_action]
                Q2_values[state][action] += ALPHA * (target - Q2_values[state][action])            
        else:
            # sarsa update (6.7)            # does use next_action for update!
            target = reward + GAMMA * Q1_values[next_state][next_action]            
            Q1_values[state][action] += ALPHA * (target - Q1_values[state][action])
                        
        state = next_state
        action = next_action
    return aleft_counts


# calculate the expected value of state over all its actions, under given epsilon greedy policy
# different states can have different number of possible actions
# action_values and action_weights are for the same state, should have the same length
def expected_value(action_values, action_weights, epsilon=EPSILON):
    best_actions = np.where(np.max(action_weights) == action_weights)[0]    
    expected_q = 0.0  # expectation is a weighted sum
    for i, a in enumerate(action_values):
        prob_a = epsilon/len(action_weights)  # section 5.4  epsilon-greedy policy
        if i in best_actions:  
            prob_a += (1.0 - epsilon)/len(best_actions)        
        expected_q += (prob_a * action_values[i])
    return expected_q

# Single, Double Expected Sarsa with epsilon-greedy policy (Ex. 6.13)
def expsarsa(Q1_values, Q2_values=None, DSL=False):
    if DSL: assert Q2_values is not None
    
    aleft_counts = 0  # number of left moves from state A
    state = STATE_A
    if DSL:
        Q_values = [q1 + q2 for q1, q2 in zip(Q1_values, Q2_values)]
        action = choose_action(Q_values, state)
    else:
        action = choose_action(Q1_values, state)            
    
    while state not in (GOAL_STATE, DECOY_STATE):
        if state == STATE_A and action == A_LEFT:
            aleft_counts += 1        
        reward, next_state = take_action(state, action) 
        # an action from S_{t+1} using Q_t values
        if DSL:
            Q_values = [q1 + q2 for q1, q2 in zip(Q1_values, Q2_values)]
            next_action = choose_action(Q_values, next_state)
        else:
            next_action = choose_action(Q1_values, next_state)        
        # S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
        if DSL:
            if rng.random() < 0.5: # (6.10)                
                # v_action = choose_action(Q1_values, next_state)
                target = reward + GAMMA * expected_value(Q2_values[next_state], Q1_values[next_state])
                Q1_values[state][action] += ALPHA * (target - Q1_values[state][action])
            else:
                # v_action = choose_action(Q2_values, next_state)
                target = reward + GAMMA * expected_value(Q1_values[next_state], Q2_values[next_state])
                Q2_values[state][action] += ALPHA * (target - Q2_values[state][action])            
        else:
            # expected sarsa update (6.9)
            target = reward + GAMMA * expected_value(Q1_values[next_state], Q1_values[next_state])            
            Q1_values[state][action] += ALPHA * (target - Q1_values[state][action])
                        
        state = next_state
        action = next_action
    return aleft_counts

#######################################################################
# Figure 6.7, 1,000 runs may be enough, 
# Number of actions in state B also affects the curves
def figure_6_7():
    num_runs = 1000
    num_episodes = 300    
    # sub-optimal move (left from A) counts
    lefts_qlearning = np.zeros((num_runs, num_episodes))
    lefts_double_qlearning = np.zeros((num_runs, num_episodes))
    lefts_sarsa = np.zeros((num_runs, num_episodes))
    lefts_double_sarsa = np.zeros((num_runs, num_episodes))
    lefts_expsarsa = np.zeros((num_runs, num_episodes))
    lefts_double_expsarsa = np.zeros((num_runs, num_episodes))
    
    for r in tqdm(range(num_runs)):
        # a list of ndarrays, initialize state-action values
        # assumes: STATE_A = 0 (2 actions), STATE_B = 1, GOAL_STATE = 2, DECOY_STATE = 3
        # Q-learning, double Q-learning
        Q_values = [np.zeros(2), np.zeros(len(B_LEFTS)), np.zeros(1), np.zeros(1)]
        Q1_values = [np.zeros(2), np.zeros(len(B_LEFTS)), np.zeros(1), np.zeros(1)]
        Q2_values = [np.zeros(2), np.zeros(len(B_LEFTS)), np.zeros(1), np.zeros(1)]
        # Sarsa, double Sarsa
        S_values = [np.zeros(2), np.zeros(len(B_LEFTS)), np.zeros(1), np.zeros(1)]
        S1_values = [np.zeros(2), np.zeros(len(B_LEFTS)), np.zeros(1), np.zeros(1)]
        S2_values = [np.zeros(2), np.zeros(len(B_LEFTS)), np.zeros(1), np.zeros(1)]
        # expSarsa, double expSarsa
        X_values = [np.zeros(2), np.zeros(len(B_LEFTS)), np.zeros(1), np.zeros(1)]
        X1_values = [np.zeros(2), np.zeros(len(B_LEFTS)), np.zeros(1), np.zeros(1)]
        X2_values = [np.zeros(2), np.zeros(len(B_LEFTS)), np.zeros(1), np.zeros(1)]
        
        for e in range(0, num_episodes):
            lefts_qlearning[r, e] = q_learning(Q_values)
            lefts_double_qlearning[r, e] = q_learning(Q1_values, Q2_values, DQL=True)
            lefts_sarsa[r, e] = sarsa(S_values)
            lefts_double_sarsa[r, e] = sarsa(S1_values, S2_values, DSL=True)
            lefts_expsarsa[r, e] = expsarsa(X_values)
            lefts_double_expsarsa[r, e] = expsarsa(X1_values, X2_values, DSL=True)            

    lefts_qlearning = lefts_qlearning.mean(axis=0)  # column wise means, runs over an episode
    lefts_double_qlearning = lefts_double_qlearning.mean(axis=0)
    lefts_sarsa = lefts_sarsa.mean(axis=0)
    lefts_double_sarsa = lefts_double_sarsa.mean(axis=0)
    lefts_expsarsa = lefts_expsarsa.mean(axis=0)
    lefts_double_expsarsa = lefts_double_expsarsa.mean(axis=0)    

    plt.plot(lefts_qlearning, label='Q-Learning')
    plt.plot(lefts_double_qlearning, label='Double Q-Learning')
    plt.plot(lefts_sarsa, label='Sarsa')
    plt.plot(lefts_double_sarsa, label='Double Sarsa')
    plt.plot(lefts_expsarsa, label='ExpSarsa')
    plt.plot(lefts_double_expsarsa, label='Double ExpSarsa')    
    plt.plot(np.ones(num_episodes) * 0.05, label='Optimal')
    plt.xlabel('episodes')
    plt.ylabel('% left actions from A')
    plt.legend()
    plt.savefig('images/my_figure_6_7.png')
    plt.close()

#######################################################################
# Q-learning eventually learns to keep right from A, but
# only after many more episodes, compared to double Q learning, 
# which is little affected by maximization bias of taking a left from A. 
# Better performance with TD control methods that use double learning.
if __name__ == '__main__':
    figure_6_7()

#######################################################################