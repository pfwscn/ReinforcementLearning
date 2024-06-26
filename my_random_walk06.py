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
# Blends ideas from MC (on-policy learning from samples) and 
# from DP (bootstrap, doesn't wait for episode completion to update)
# TD values are estimates in two senses: estimates of sample returns
#
# constant-alpha MC: V(S_t) = V(S_t) + alpha * [G_t - V(S_t)] (6.1)
#   update values at the end of an episode when the return G_t (actual sample return) is known
#
# one-step TD, TD(0): V(S_t) = V(S_t) + alpha * [R_t+1 + gamma * V(S_t+1) - V(S_t)] (6.2)
#   update values after one time-step when actual reward R_t+1 and current estimate V(S_t+1) are known
#
# Backup diagram for tabular TD(0): no branching like MC backup diagram, 
#   but unlike MC need not end in terminal state (number of steps depends on TD error).
# TD error: delta_t = R_t+1 + gamma * V(S_t+1) - V(S_t) (6.5) 
#   is the error in V(S_t) known at t+1 (after an action has been taken at timestep t)
#
# Both MC and TD do sample updates (based on a single sample trajectory); 
#   DP does expected updates (branching in DP backup diagram reflects env. dynamics)
# Convergence (to the correct value) guarantees for both MC and TD.
#
# Section 6.3: Optimality of TD(0) (via batch-updating)
# A method to re-use limited amount of sample/training data to estimate/predict state-values.
# State-values updated once per batch, and then used as initial state-values for the next batch;
#   repeat until convergence.
# Increments are computed after every episode (6.1 MC), or 
#   for every time-step at which a nonterminal state is visited (6.2 TD), but 
#   the value function is changed only once, by the sum of all increments,
#   after processing each complete batch of training data.
# "Under batch updating, TD(0) converges deterministically to a single answer 
#   independent of the step-size parameter, alpha, as long as alpha is chosen to be
#   sufficiently small. The constant-alpha MC also converges deterministically under    
#   the same conditions, but to a different answer."
# * Normal (non-batch) updating move towards, but not all the way, to batch answers (?)
# Example 6.4 
#   Batch MC methods find estimates that minimize mean square error on the training set
#   (lower error on existing/present data);
#   batch TD(0) find estimates that are most likely to generate the training set 
#   (constructs a maximum-likelihood model of the process, lower error on future data).
# "In general, batch TD(0) converges to the certainty-equivalence estimate."
# "In batch form, TD(0) is faster than MC because it computes the true 
#   certainty-equivalence estimate."  
# "On tasks with large state spaces, TD methods may be the only feasible way of
#   approximating the certainty-equivalence solution."
#
# Without a model, predicting (policy evaluation) action-values instead of 
# state-values is more useful for policy improvement (Control), (Section 5.2). 
# The goal of RL is to learn the optimal (pi*) or move towards the best policy,
# that will guide future actions.
#
# Section 6.4: Sarsa On-policy TD control (windy grid)
# Section 6.5: Q-learning Off-policy TD control (cliff walking)
#
#######################################################################

import numpy as np
from tqdm import tqdm
rng = np.random.default_rng()

# Example 6.2
# Generally, TD methods found to converge faster than constant-alpha MC methods on stochastic tasks
# (section 6.3 certainty-equivalence estimate)
# Markov reward process: (0)Z <- A <-> B <-> C <-> D <-> E -> F(1)
# Always starts at C, and terminates eiher at the extreme left (return 0) or extreme right (return 1)
# Actions (left, right) are equiprobable (0.5); reward 0 for all except E->F
# Undiscounted task (gamma=1.0)
# True values of each state is the probability of each state A...E reaching F
# Walks are Markov, fresh-start property: doesn't matter where they started or continue from
#

TRUE_VALUES = [0, 1/6, 2/6, 3/6, 4/6, 5/6, 1]
num_episodes = 1000000  # 1,000,000
def get_true_values_A():    
    state_values = np.zeros(7, dtype=float)  # Z(0), A(1), B(2), C(3), D(4), E(5), F(6)
    # Walks starting in Z have 0 probability to terminate in F
    state_values[6] = 1.0  # Walks starting in F terminates in F, so probability=1.0
    for s in range(1, 6):        
        success_counts = 0        
        for e in range(0, num_episodes, 1):  # episodes, an episode may take a long time to terminate!
            current_state = s
            while True:
                if rng.random() < 0.5:
                    current_state = current_state - 1  # move left if possible
                    if current_state < 1:                        
                        break
                else:
                    current_state = current_state + 1  # move right if possible
                    if 6 == current_state:
                        success_counts += 1                        
                        break
        print(s, success_counts, e)
        state_values[s] = success_counts/num_episodes  # frequentist
    rms = np.sqrt(np.mean(np.power(state_values - TRUE_VALUES, 2)))
    print('RMS ', rms)
    print(state_values) # similar results as methods B and C below
    print(TRUE_VALUES, '\n')

# First-visit (B) and Every-visit (C) give similar results
def get_true_values_B():       
    state_values = np.zeros(7, dtype=float)  # Z(0), A(1), B(2), C(3), D(4), E(5), F(6)
    # Walks starting in Z have 0 probability to terminate in F
    state_values[6] = 1.0  # Walks starting in F terminates in F, so probability=1.0    
    state_visits = np.zeros(7, dtype=int)
    state_visits[0] = 1; state_visits[6] = 1  # avoid division by 0    
    # for e in range(0, num_episodes, 1):  # episodes, an episode may take a long time to terminate!
    for e in tqdm(range(0, num_episodes, 1)):
        state = 3; walk = [state]
        while True:
            if rng.random() < 0.5:              
                state = state - 1  # move left if possible
                if state < 1:                    
                    state_visits[walk] += 1  # first-visit: multiple visits to a state will count as one!
                    break  # start another episode
                else:
                    walk.append(state)
            else:
                state = state + 1  # move right if possible
                if state < 6:
                    walk.append(state)
                else:  # state == 6, reward all (unique!) states in this successful walk
                    state_values[walk] += 1
                    state_visits[walk] += 1  
                    break    
    state_values[1:6] /= state_visits[1:6]
    rms = np.sqrt(np.mean(np.power(state_values - TRUE_VALUES, 2)))
    print('RMS ', rms)
    print(state_values)
    print(TRUE_VALUES, '\n')

def get_true_values_C():
    state_values = np.zeros(7, dtype=float)  # Z(0), A(1), B(2), C(3), D(4), E(5), F(6)
    # Walks starting in Z have 0 probability to terminate in F
    state_values[6] = 1.0  # Walks starting in F terminates in F, so probability=1.0
    state_visits = np.zeros(7, dtype=int)
    state_visits[0] = 1; state_visits[6] = 1  # avoid division by 0
    # for e in range(0, num_episodes, 1):  # episodes, an episode may take a long time to terminate!
    for e in tqdm(range(0, num_episodes, 1)):
        state = 3; walk = [state]
        while True:
            if rng.random() < 0.5:              
                state = state - 1  # move left if possible                
                if state < 1:
                    # every-visit: make visits to a state count each time
                    for s in range(1, 6): 
                        state_visits[s] += walk.count(s)
                    break  # start another episode
                else:
                    walk.append(state)
            else:
                state = state + 1  # move right if possible                                
                if state > 5:  
                    # reward all middle states in this successful walk
                    for s in range(1, 6): 
                        state_values[s] += walk.count(s)
                        state_visits[s] += walk.count(s)
                    break
                else:
                    walk.append(state)    
    state_values[1:6] /= state_visits[1:6]
    rms = np.sqrt(np.mean(np.power(state_values - TRUE_VALUES, 2)))
    print('RMS ', rms) 
    print(state_values)
    print(TRUE_VALUES, '\n')

#######################################################################
# Example 6.2: Compare TD(0) with constant-alpha MC
    
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GAMMA = 1.0
def temporal_difference(state_values, alpha=0.1, batch=False):    
    state = 3
    walk = [state]; rewards = []    # walk includes first and last/terminal state
    while state not in [0, 6]:      # Terminal state
        if rng.random() < 0.5:
            next_state = state - 1  # left
        else:
            next_state = state + 1  # right
        # take action, collect reward, identify next state, update current state-value
        # go to next state, repeat until reach terminal state
        reward = 0  # reward for all transitions are 0
        rewards.append(reward)  
        # TD(0) sample update: update state_values (V) on every step if not batch updating
        if not batch:
            target = reward + GAMMA * state_values[next_state] 
            state_values[state] += alpha * (target - state_values[state])
        walk.append(next_state)
        state = next_state
    return walk, rewards

def monte_carlo(state_values, alpha=0.1, batch=False):
    state = 3
    walk = [state]; returns = []  # walk includes first and last/terminal states
    # experience an episode, a complete walk from state C(3)
    while state not in [0, 6]:
        if rng.random() < 0.5:
            state -= 1  # left; no need for current and next states
        else:
            state += 1  # right
        # MC: no reward or updates until the end of an episode
        walk.append(state)
    if 6 == state:  # return G_T = 1 if right-most state reached, 0 otherwise, for all visited states
        returns = [1] * (len(walk) - 1)
    else:
        returns = [0] * (len(walk) - 1)
    # MC update state_values
    if not batch:
        for s, r in zip(walk[:-1], returns):  # backup the return for all states visited in this episode
            state_values[s] += alpha * (r - state_values[s])
    return walk, returns

INITIAL_STATE_VALUES = np.full(7, 0.5, dtype=np.float64)  # initial non-terminating state values = 0.5
INITIAL_STATE_VALUES[[0, -1]] = [0, 1]  # left-most and right-most state values = 0, 1

# Example 6.2 left
# TD(0) policy evaluation, prediction (estimate state-value) problem
def estimate_state_value(alpha):
    episodes = [0, 1, 10, 100, 200, 500]
    state_values = np.copy(INITIAL_STATE_VALUES)
    plt.figure(1)
    for i in range(episodes[-1] + 1):
        if i in episodes:
            rms = np.sqrt( np.sum(np.power(state_values - TRUE_VALUES, 2)) /5.0 )
            print(i, rms)
            plt.plot(("A", "B", "C", "D", "E"), state_values[1:6], label=str(i) + ' episodes %.03f' % (rms))
        temporal_difference(state_values, alpha)   # update state_values with TD(0)
    plt.plot(("A", "B", "C", "D", "E"), TRUE_VALUES[1:6], label='true values')
    plt.xlabel('State')
    plt.ylabel('Estimated Value TD(0) alpha=' + str(alpha))
    plt.legend()

# Example 6.2 right (Learning curves)
# How estimated/predicted state_values approach their true values, with different step_sizes (alpha)
# Observe behavior of rms_error (averaged over 100 runs) as a function of number of episodes
# TD(0) state-values updated within an episode; MC state-values updated after an episode
# Step-size (alpha) determines size of correction/update to current state-value: 
#   V(S_t) = V(S_t) + alpha * [target - V(S_t)]
# Learning with smaller alpha takes longer, but yields better results (stable & smaller rms) in the long run.
# Case for non-constant alpha
def rms_error():
    td_alphas = [0.15, 0.1, 0.05, 0.01]
    mc_alphas = [0.01, 0.02, 0.03, 0.04]
    num_episodes = 500
    num_runs = 100
    for i, alpha in enumerate(td_alphas + mc_alphas):
        # accumulated rms_errors per (method, alpha) pair, over 100 episodes in 100 runs
        total_errors = np.zeros(num_episodes)  
        if i < len(td_alphas):
            method = 'TD'
            linestyle = 'solid'
        else:
            method = 'MC'
            linestyle = 'dashdot'
        for r in tqdm(range(num_runs)):
            errors = []  # (200, ) rms_errors over 200 episodes in a run
            state_values = np.copy(INITIAL_STATE_VALUES)
            for i in range(0, num_episodes):  
                # first and last elements always agree, consider only the 5 non-terminal states for rms
                rms = np.sqrt( np.sum(np.power(state_values - TRUE_VALUES, 2)) /5.0 )
                errors.append(rms)
                if method == 'TD':
                    temporal_difference(state_values, alpha)
                else:
                    monte_carlo(state_values, alpha)
            total_errors += np.asarray(errors)
        total_errors /= num_runs
        plt.plot(total_errors, linestyle=linestyle, label=method + ', $\\alpha$ = %.02f' % (alpha))
    plt.xlabel('Walks/Episodes')
    plt.ylabel('Empirical RMS error, averaged over states')
    plt.legend(); plt.grid()

def example_6_2():
    plt.figure(figsize=(10, 20))
    plt.subplot(3, 1, 1)
    estimate_state_value(0.1)
    plt.subplot(3, 1, 2)
    estimate_state_value(0.01)

    plt.subplot(3, 1, 3)
    rms_error()
    plt.tight_layout()
    plt.savefig('images/my_example_6_2.png')
    plt.close()

#######################################################################
# Figure 6.2
# method is either 'TD' or 'MC', alpha is kept small (section 6.3)

def batch_updating(method, num_episodes, alpha=0.001): 
    num_runs = 100
    total_errors = np.zeros(num_episodes)
    for r in tqdm(range(0, num_runs)):
        state_values = np.copy(INITIAL_STATE_VALUES)
        # state_values[1:6] = -1 # ??
        errors = []
        all_walks = []; all_rewards = []
        for e in range(num_episodes):  
            # add an episode to current data
            # with batch-updating, methods merely return rewards and do not update state_values
            if 'TD' == method:  
                walks, rewards = temporal_difference(state_values, batch=True)
            else:
                walks, rewards = monte_carlo(state_values, batch=True)
            all_walks.append(walks); all_rewards.append(rewards)
            # keep batch-updating state-values with data observed so far, until state-values converge (updates -> 0)
            # an episode is reused multiple times (earlier episodes more influence on results?)
            prev = 0.0
            while True:   
                state_updates = np.zeros(7)  # accumulate update error for each state
                for walk, rewards in zip(all_walks, all_rewards):  # process a batch of episodes
                    for i in range(0, len(walk) - 1):  
                        p = walk[i]; r = rewards[i]; q = walk[i+1]  # S_t, R_{t+1}, S_{t+1}
                        if 'TD' == method:
                            state_updates[p] += r + state_values[q] - state_values[p]
                        else:
                            state_updates[p] += r - state_values[p]
                state_updates *= alpha
                # perform batch updating
                state_values += state_updates                
                curr = np.sum(np.abs(state_updates))
                # if np.sum(np.abs(state_updates)) < 1e-3:  # will not break if process is not converging (alpha too big)!
                if curr < 1e-3 or curr > prev + 1e-3:
                    break
                prev = curr
            # before observing new data, calculate rms error of current state_values; mean is over 5 not 7 states
            errors.append(np.sqrt (np.sum(np.power(state_values - TRUE_VALUES, 2)) / 5.0) )        
        total_errors += np.asarray(errors)  # over runs
    total_errors /= num_runs  # rms error of state_values estimated by batches of episodes, averaged over runs
    return total_errors


def figure_6_2():
    num_episodes = 100
    
    alpha = 0.001    
    # alpha = 0.01
    
    td_errors = batch_updating('TD', num_episodes, alpha)
    mc_errors = batch_updating('MC', num_episodes, alpha)

    plt.plot(td_errors, label='TD')
    plt.plot(mc_errors, label='MC')
    plt.title("Batch Training")
    plt.xlabel('Walks/Episodes')
    plt.ylabel('RMS error of state-values, averaged over 100 runs')
    plt.xlim(0, 100)
    plt.ylim(0, 0.25)
    plt.legend()
    plt.grid()
    plt.savefig('images/my_figure_6_2_' + str(alpha) + '.png')
    plt.close()

#######################################################################
if __name__ == '__main__':
    
    # get_true_values_A()  # Ex 6.6
    # get_true_values_B()
    # get_true_values_C()
    
    example_6_2()    
    figure_6_2()
    
#######################################################################