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
#
# RL is learning what to do, from (limited) experience.
#   State or action values quantify past experiences, to guide future action decisions.
#   Getting action values right is the essence of RL.
#
# Estimate action-values; transitions are between state-action pairs (s, a) -> (s', a')
#   Backup diagram starts with action node, and ends with action node
#
# General form of sample update: Q(S_t, A_t) += alpha * [target - Q(S_t, A_t)]
#
# Section 6.4: Sarsa On-policy TD control
# epsilon-greedy policy applied to choose A_t from S_t, and 
#   also to choose A_t+1 from S_t+1 (for value update and action from next state)
# target = R_t+1 + gamma * Q(S_t+1, A_t+1) (6.7)
# No branching in backup diagram 
#
# Section 6.5 Q-Learning Off-policy TD control
# "learned action-value function Q directly approximates q*, the optimal action-value function,
# independent of the policy being followed. ... policy still ... determines which state-action 
# pairs are visited and updated."
# epsilon-greedy policy applied to choose A_t from S_t, and 
#   greedy policy chooses A_t+1 from S_t+1 (for value update only, not action from next state)
#   epsilon-greedy policy still applied to choose A_t+1 from S_t+1 
# target = R_t+1 + gamma * max_a Q(S_t+1, A_t+1=a) (6.8)
# Effectively also no branching in backup diagram; max taken over action nodes in next state
# Convergence
# Ex. 6.11 Q-learning is considered an off-policy method because learning is from data
# generated under a policy different from the one used to update action-values (section 5.5).
# 
# Section 6.6 Expected Sarsa
# Like Q-learning except instead of max over next state action values (for value update),
#   expected value (of state) is used, with probability of action under (behavior) policy. 
# Expected Sarsa (off-policy TD control): instead of using the policy to choose A_{t+1} (Sarsa 6.7), 
#   or taking the max of existing Q(S_{t+1} values over all actions (Q-learning 6.8), 
#   to update Q(S_t), it uses expectation of existing Q(S_{t+1} values over all actions (6.9).
#   Behavior policy still used to choose actions in episodes.
# "eliminates the variance due to (Sarsa's) random selection of A_t+1"
# target = R_t+1 + gamma * E_pi[ Q(S_t+1, A_t+1) | S_t+1 ] (6.9)
# "Given the next state S_t+1, this algo moves deterministically in the same direction as 
# Sarsa moves in expectation."
# Effectively also no branching in backup diagram; weighted avg taken over action nodes in next state
# Convergence and performance
#
#######################################################################

import numpy as np
# import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

rng = np.random.default_rng()

# Example 6.6 Cliff walking
# Compares Sarsa (on-policy) with Q-learning (off-policy)
# Expected Sarsa is on-policy, in this example

GAMMA = 1.0     # Undiscounted episodic task

# Top-leftmost cell is [0, 0]
GRID_HEIGHT, GRID_WIDTH = 4, 12
START_CELL = [3, 0]
GOAL_CELL = [3, 11]

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTION_FIGS=['U', 'D', 'L', 'R']
NUM_ACTIONS = len(ACTIONS)

# Rewards are -1 on all transitions, except those on the 'Cliff' region [(3, 1) ... (3, 10)]
# Stepping into the Cliff incurs a penalty of 100 (Reward = -100), and 
# transports the agent back to the start.
def step(state, action):
    i, j = state
    if action == UP:
        next_state = [max(i - 1, 0), j]
    elif action == LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == RIGHT:
        next_state = [i, min(j + 1, GRID_WIDTH - 1)]
    elif action == DOWN:
        next_state = [min(i + 1, GRID_HEIGHT - 1), j]
    else:
        assert False    
    # the Cliff region
    if 3 == next_state[0] and 0 < next_state[1] < 11:  
        reward = -100
        next_state = START_CELL
    else:
        reward = -1
    return next_state, reward

# epsilon-greedy policy; when epsilon=0.0, a best (max) action is choosen
def choose_action(Q_values, state, epsilon):
    if rng.random() < epsilon:
        action = rng.choice(ACTIONS, shuffle=False)
    else:        
        action_values = Q_values[state[0], state[1], :]
        best_actions = np.where(np.max(action_values) == action_values)[0]
        action = rng.choice(best_actions.tolist(), shuffle=False)
    return action

# a Sarsa (on-policy)  episode
# Q_values: state-action pair values, will be updated
# expected: use expected Sarsa if True
# returns total reward for this episode
def trek_sarsa(Q_values, epsilon, alpha, expected=False):
    # indicates the type of path taken (length, entered cliff region)
    # longer paths, those entered cliff region with have smaller rewards (larger negative value)
    total_reward = 0.0
    state = START_CELL
    action = choose_action(Q_values, state, epsilon)
    while state != GOAL_CELL:
        next_state, reward = step(state, action)    # S_t, A_t, R_t+1, S_t+1        
        next_action = choose_action(Q_values, next_state, epsilon)   # A_t+1 for value update and next action
        total_reward += reward

        if not expected:            
            target = reward + GAMMA * Q_values[next_state[0], next_state[1], next_action]
        else:
            # calculate the expected value of new state over all actions            
            action_values = Q_values[next_state[0], next_state[1], :]
            best_actions = np.where(np.max(action_values) == action_values)[0]            
            # best_actions = np.argwhere(action_values == np.max(action_values)).flatten()             
            expected_q = 0.0  # for the next_state
            for a in ACTIONS:
                if a in best_actions:  # section 5.4  epsilon-greedy policy
                    prob_a = (1.0 - epsilon)/len(best_actions) + epsilon/NUM_ACTIONS
                else:
                    prob_a = epsilon/NUM_ACTIONS
                # expected_q += (prob_a * Q_values[next_state[0], next_state[1], a])  # expectation, weighted sum
                expected_q += (prob_a * action_values[a])
            target = reward + GAMMA * expected_q

        # (Expected) Sarsa update
        Q_values[state[0], state[1], action] += alpha * (target - Q_values[state[0], state[1], action])        
        state = next_state
        action = next_action
    return total_reward

# a Q-learning episode
# Q_values: state-action pair values, will be updated
# returns total reward for this episode
def trek_qlearning(Q_values, epsilon, alpha):
    total_reward = 0.0
    
    state = START_CELL    
    while state != GOAL_CELL:
        action = choose_action(Q_values, state, epsilon)
        next_state, reward = step(state, action)    # S_t, A_t, R_t+1, S_t+1
        total_reward += reward
        target = reward + GAMMA * np.max(Q_values[next_state[0], next_state[1], :])  # max A_t+1 for value update only
        
        # Q-Learning update
        Q_values[state[0], state[1], action] += alpha * (target - Q_values[state[0], state[1], action])
        state = next_state
    return total_reward

# off-policy exp sarsa
# behavior is epsilon-greedy (epsilon > 0.0), target policy is greedy
# def trek_expsarsa(Q_values, epsilon, alpha):
#     total_reward = 0.0
#    
#     state = START_CELL    
#     while state != GOAL_CELL:
#         action = choose_action(Q_values, state, epsilon)
#         next_state, reward = step(state, action)    # S_t, A_t, R_t+1, S_t+1
#         total_reward += reward
#         # calculate the expected value of new state over all actions            
#         action_values = Q_values[next_state[0], next_state[1], :]
#         # best_actions = np.where(np.max(action_values) == action_values)[0]
#         best_actions = np.argwhere(action_values == np.max(action_values)).flatten()             
#         expected_q = 0.0  # for the next_state
#         # if target policy is greedy (epsilon=0), expectation is over best actions only 
#         # since prob. of other actions = 0.0; best actions all have the same value, so
#         # equivalent to q-learning
#         for a in ACTIONS:   
#             if a in best_actions:  # section 5.4  epsilon-greedy policy
#                 prob_a = (1.0 - epsilon)/len(best_actions) + epsilon/NUM_ACTIONS
#             else:
#                 prob_a = epsilon/NUM_ACTIONS  # 0.0
#             # expected_q += (prob_a * Q_values[next_state[0], next_state[1], a])  # expectation, weighted sum
#             expected_q += (prob_a * action_values[a])
#         target = reward + GAMMA * expected_q
#         # (Expected) Sarsa update
#         Q_values[state[0], state[1], action] += alpha * (target - Q_values[state[0], state[1], action])        
#         state = next_state        
#     return total_reward

#######################################################################
# print optimal policy
def print_optimal_policy(Q_values):
    optimal_policy = []
    for i in range(0, GRID_HEIGHT):
        optimal_policy.append([])
        for j in range(0, GRID_WIDTH):
            if [i, j] == GOAL_CELL:
                optimal_policy[-1].append('G')
                continue
            best_action = choose_action(Q_values, [i, j], 0.0)
            optimal_policy[-1].append(ACTION_FIGS[best_action])
    for row in optimal_policy:
        print(row)
 
# Average results over a sliding window of size 100, across a single run of 1000 episodes, 
# instead of doing multiple runs
def figure_6_4(epsilon, alpha):
    num_runs = 10 #100
    num_episodes = 1000
    # collect total reward per episode; 
    sarsa_rewards = np.zeros(num_episodes)
    expsarsa_rewards = np.zeros(num_episodes)
    qlearning_rewards = np.zeros(num_episodes)
    for r in tqdm(range(num_runs)):
        sarsa_Qvalues = np.zeros((GRID_HEIGHT, GRID_WIDTH, NUM_ACTIONS))
        expsarsa_Qvalues = np.zeros((GRID_HEIGHT, GRID_WIDTH, NUM_ACTIONS))
        qlearning_Qvalues = np.zeros((GRID_HEIGHT, GRID_WIDTH, NUM_ACTIONS))
        for e in range(0, num_episodes):
            sarsa_rewards[e] += trek_sarsa(sarsa_Qvalues, epsilon, alpha)
            expsarsa_rewards[e] += trek_sarsa(expsarsa_Qvalues, epsilon, alpha, expected=True)
            qlearning_rewards[e] += trek_qlearning(qlearning_Qvalues, epsilon, alpha)

    win_sz = 100
    slide_win = np.ones(win_sz)
    sarsa_res = np.convolve(sarsa_rewards, slide_win, 'valid')/win_sz
    expsarsa_res = np.convolve(expsarsa_rewards, slide_win, 'valid')/win_sz
    qlearning_res = np.convolve(qlearning_rewards, slide_win, 'valid')/win_sz

    # draw reward curves
    plt.plot(sarsa_res, label='Sarsa')
    plt.plot(expsarsa_res, label='Expected Sarsa')
    plt.plot(qlearning_res, label='Q-Learning')
    plt.title('epsilon ' + str(epsilon) + ' alpha ' + str(alpha))
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')    
    plt.legend()
    # 1 runs:1000 episodes
    plt.savefig('images/my_figure_6_4_eps_' + str(epsilon) + '_alpha_' + str(alpha) + '.png')    
    # 100 runs: 100 * 1000 episodes
    # plt.savefig('images/my_figure_6_4a_eps_' + str(epsilon) + '_alpha_' + str(alpha) + '.png')
    plt.close()

    # display optimal policy
    print('Sarsa Optimal Policy:')
    print_optimal_policy(sarsa_Qvalues)
    print('Expected Sarsa Optimal Policy:')
    print_optimal_policy(expsarsa_Qvalues)    
    print('Q-Learning Optimal Policy:')
    print_optimal_policy(qlearning_Qvalues)


def figure_6_6():
    EPSILON = 0.1
    ALPHAS = np.arange(0.1, 1.1, 0.1)
    num_alphas = len(ALPHAS)
    # methods    
    asymtotic_SARSA = 0 
    asymtotic_EXPSARSA = 1
    asymtotic_QLEARNING = 2
    interim_SARSA = 3
    interim_EXPSARSA = 4
    interim_QLEARNING = 5        
    
    results = np.zeros((6, num_alphas))
    num_runs = 10
    num_episodes = 2000
    for r in range(num_runs):
        for a, alpha in enumerate(ALPHAS):
            sarsa_Qvalues = np.zeros((GRID_HEIGHT, GRID_WIDTH, NUM_ACTIONS))
            expsarsa_Qvalues = np.zeros((GRID_HEIGHT, GRID_WIDTH, NUM_ACTIONS))
            qlearning_Qvalues = np.zeros((GRID_HEIGHT, GRID_WIDTH, NUM_ACTIONS))
            
            for e in tqdm(range(num_episodes)):
                sarsa_reward = trek_sarsa(sarsa_Qvalues, epsilon=EPSILON, alpha=alpha, expected=False)
                expsarsa_reward = trek_sarsa(expsarsa_Qvalues, epsilon=EPSILON, alpha=alpha, expected=True)
                qlearning_reward = trek_qlearning(qlearning_Qvalues, epsilon=EPSILON, alpha=alpha)
                
                results[asymtotic_SARSA, a] += sarsa_reward
                results[asymtotic_EXPSARSA, a] += expsarsa_reward
                results[asymtotic_QLEARNING, a] += qlearning_reward

                if e < 100:
                    results[interim_SARSA, a] += sarsa_reward
                    results[interim_EXPSARSA, a] += expsarsa_reward
                    results[interim_QLEARNING, a] += qlearning_reward

    results[3:, :] /= (100 * num_runs)  # interim average
    results[:3, :] /= (num_episodes * num_runs)  # asymptotic average
    
    labels = ['Asymptotic Sarsa', 'Asymptotic Expected Sarsa', 'Asymptotic Q-Learning',
              'Interim Sarsa', 'Interim Expected Sarsa', 'Interim Q-Learning']
    for m, label in enumerate(labels):
        plt.plot(ALPHAS, results[m, :], label=label)    
    plt.xlabel('alpha')
    plt.ylabel('reward per episode')
    plt.legend()
    plt.savefig('images/my_figure_6_6.png')
    plt.close()

#######################################################################
# Fig. 6.4
# Sarsa takes the longer safer path, while Q-Learning takes shorter riskier path
# Q-learning performs worse than Sarsa because it is not cushioned against occasional falls
# into the cliff region by actions of the epsilon-greedy policy.
# With smaller epsilon, this performance difference decreases.
# When (behavior) policies become more greedy, performance difference less distinguishable.
# Expected Sarsa takes the middle road, and ends up with better total_rewards (less negative), 
# but depends on alpha. With smaller alpha, difference between Expected Sarsa and Sarsa decreases.
#
# Fig. 6.6 Alpha can be large for Expected Sarsa (& Q-learning), but not for Sarsa. 
#
# "In cliff walking, the state transitions are all deterministic and all randomness comes from
# the policy. In such cases, Expected Sarsa can safely set alpha=1 without suffering any
# degradation of asymptotic performance, whereas Sarsa can only perform well in the long run
# at a small value of alpha, at which short-term performance is poor. "
# "there is a consistent empirical advantage of Expected Sarsa over Sarsa."
#
# "suppose pi is the greedy policy while behavior is more exploratory; then Expected Sarsa is
# exactly Q-learning. In this sense Expected Sarsa subsumes and generalizes Q-learning
# while reliably improving over Sarsa."   

if __name__ == '__main__':
    figure_6_4(0.1, 0.5)    # epsilon, alpha
    figure_6_4(0.1, 0.1)    
    figure_6_4(0.01, 0.5)
    figure_6_4(0.0, 0.5)    # Ex. 6.12 all policies are greedy
    
    figure_6_6()
    
#######################################################################