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
# Section 6.4: Sarsa On-policy TD control
# Learn action-values q_pi(s, a), instead of state-values v_pi(s)
# Transitions are between state-action pairs (s, a) -> (s', a')
# Q(S_t, A_t) += alpha * [R_t+1 + gamma * Q(S_t+1, A_t+1) - Q(S_t, A_t)] (6.7)
#   Q(S_T, A_T) = 0
# Backup diagram starts with action node, no branching
# Continually estimate q_pi for the target/behavior policy pi, 
#   while making pi more greedy with respect to q_pi.
# Convergence properties of Sarsa method: alpha, policy, 
#   sampling (exploration)
# Example 6.5 Windy gridworld: undiscounted episodic task
# "MC methods cannot easily be used here because termination is not guaranteed
# for all policies... Online learning methods such as Sarsa do not suffer this
# problem because they learn during the episode that such policies are poor
# and switch to someting else."
#
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rng = np.random.default_rng()

# Ex 6.5 Windy Gridworld

# Top-leftmost cell is [0, 0]
GRID_HEIGHT, GRID_WIDTH = 7, 10
# Wind blows upward per column, affects actions taken from a cell
WIND_STRENGTH = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]
# ACTIONS_FIGS=['↑', '↓', '←', '→']
ACTION_FIGS=['U', 'D', 'L', 'R']

# Step or move made in the direction of action, in the windy gridworld
def move(state, action):
    i, j = state
    if UP == action:
        return [max(i - 1 - WIND_STRENGTH[j], 0), j]
    elif DOWN == action:
        return [max(min(i + 1 - WIND_STRENGTH[j], GRID_HEIGHT - 1), 0), j]
    elif LEFT == action:
        return [max(i - WIND_STRENGTH[j], 0), max(j - 1, 0)]
    elif RIGHT == action:
        return [max(i - WIND_STRENGTH[j], 0), min(j + 1, GRID_WIDTH - 1)]
    else:
        assert False

# undiscounted, episodic task with constant rewards -1 until goal is reached
# on-policy: epsilon-greedy to maintain some level of exploration

START_STATE = [3, 0]
GOAL_STATE = [3, 7]    
GAMMA = 1.0
REWARD = -1.0  # reward for each action

# Make a trek through the windy grid (an episode from a start state, to hopefully reach the goal state)
def trek(q_values, initial_state, epsilon, alpha):    

    # policy is epsilon-greedy    
    def choose_action(state):
        if rng.random() < epsilon:
            action = rng.choice(ACTIONS, shuffle=False)
        else:
            action_values = q_values[state[0], state[1], :]
            best_actions = np.where(np.max(action_values) == action_values)[0]
            action = rng.choice(best_actions.tolist(), shuffle=False)
            # action = np.random.choice([a for a, v in enumerate(action_values) if v == np.max(action_values)])[0]
        return action

    # state = START_STATE 
    # random start state (exploring starts); but using stochastic policy
    # state = [rng.integers(0, GRID_HEIGHT), rng.integers(0, GRID_WIDTH)]
    state = initial_state
    actions = [] # track the moves taken in this episode
    action = choose_action(state)
    actions.append(action)
    # keep going until get to the goal state
    while state != GOAL_STATE:
        next_state = move(state, action)
        next_action = choose_action(next_state)
        # Sarsa update
        q_values[state[0], state[1], action] += \
            alpha * (REWARD + GAMMA * q_values[next_state[0], next_state[1], next_action] -
                        q_values[state[0], state[1], action])
        state = next_state
        action = next_action
        actions.append(action)        
    return actions, len(actions)

#######################################################################
# as the agent learns, time to reach the goal (number of steps per episode) should decrease to optimal (15)
def figure_6_3a():
    Q_values = np.zeros((GRID_HEIGHT, GRID_WIDTH, len(ACTIONS)))  # for each (s, a) pair    
    num_episodes = 500
    steps = []  # number of steps per episode (trek)
    for e in range(num_episodes):
        if e < 250:
            random_state = [rng.integers(0, GRID_HEIGHT), rng.integers(0, GRID_WIDTH)]  # exploring-starts
            actions, num_steps = trek(Q_values, random_state, epsilon=0.1, alpha=0.5) 
        else:
            actions, num_steps = trek(Q_values, START_STATE, epsilon=0.01, alpha=0.5)   # more greedy
            steps.append(num_steps)
            if num_steps < 17:
                print(e, num_steps, [ ACTION_FIGS[a] for a in actions ] )
    cum_steps = np.add.accumulate(steps)  # cumulative sum of steps
    avg_steps = cum_steps/np.arange(1, len(steps)+1)
    
    num_episodes = len(steps)
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(1, num_episodes+1), steps)
    plt.plot(np.arange(1, num_episodes+1), avg_steps)
    plt.plot(np.arange(1, num_episodes+1), [17]*num_episodes)
    plt.xlabel('Episodes')
    plt.ylabel('Time steps') 
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(cum_steps, np.arange(1, num_episodes+1))    
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.grid()
    
    # plt.subplot(3, 1, 3)
    # table to display the optimal policy
    
    plt.tight_layout()
    plt.savefig('images/my_figure_6_3a.png')
    plt.close()
    
    # display the optimal policy
    optimal_policy = []
    for i in range(0, GRID_HEIGHT):
        optimal_policy.append([])  # add a new empty row
        for j in range(0, GRID_WIDTH):
            action_values = Q_values[i, j, :]
            best_actions = np.where(np.max(action_values) == action_values)[0]
            pol=''
            for a in best_actions:
                pol += ACTION_FIGS[a]
            # add state labels
            if [i, j] == GOAL_STATE:
                pol = str(pol) + " G"
            elif [i, j] == START_STATE:
                pol = str(pol) + " S"
            else:
                pol = str(pol) + "  "
            optimal_policy[-1].append(pol)  # append to the last row
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)
    print('Wind strength for each column:\n{}'.format([str(w) for w in WIND_STRENGTH]))

#######################################################################
# as the agent learns, time to reach the goal (number of steps per episode) should decrease to optimal (15)
def figure_6_3b():
    Q_values = np.zeros((GRID_HEIGHT, GRID_WIDTH, len(ACTIONS)))  # for each (s, a) pair    

    random_state = [rng.integers(0, GRID_HEIGHT), rng.integers(0, GRID_WIDTH)]  # exploring-starts
    actions, num_steps = trek(Q_values, random_state, epsilon=0.1, alpha=0.5) 
    num_episodes = 1
    prev_qvalues = Q_values.copy()
    while True:
        random_state = [rng.integers(0, GRID_HEIGHT), rng.integers(0, GRID_WIDTH)]  # exploring-starts
        actions, num_steps = trek(Q_values, random_state, epsilon=0.1, alpha=0.5) 
        num_episodes += 1
        if np.sum(np.abs(Q_values - prev_qvalues)) < 1e-6:
            break
        else:
            prev_qvalues = Q_values.copy()
    print(num_episodes)    
    
    steps = []  # number of steps per episode (trek)
    cum_steps, avg_steps = [], []    
    while True:
        actions, num_steps = trek(Q_values, START_STATE, epsilon=0.001, alpha=0.5)   # more greedy
        steps.append(num_steps)
        cum_steps = np.add.accumulate(steps)  # cumulative sum of steps
        avg_steps = cum_steps/np.arange(1, len(steps)+1)
        # print(avg_steps[-1])
        if avg_steps[-1] < 17:
            print(avg_steps[-1], len(steps))
            break;
    
    num_episodes = len(steps)
    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(1, num_episodes+1), steps)
    plt.plot(np.arange(1, num_episodes+1), avg_steps)
    plt.plot(np.arange(1, num_episodes+1), [17]*num_episodes)
    plt.xlabel('Episodes')
    plt.ylabel('Time steps') 
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(cum_steps, np.arange(1, num_episodes+1))    
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    plt.grid()
    
    # plt.subplot(3, 1, 3)
    # table to display the optimal policy
    
    plt.tight_layout()
    plt.savefig('images/my_figure_6_3b.png')
    plt.close()
    
    # display the optimal policy
    optimal_policy = []
    for i in range(0, GRID_HEIGHT):
        optimal_policy.append([])  # add a new empty row
        for j in range(0, GRID_WIDTH):
            action_values = Q_values[i, j, :]
            best_actions = np.where(np.max(action_values) == action_values)[0]
            pol=''
            for a in best_actions:
                pol += ACTION_FIGS[a]
            # add state labels
            if [i, j] == GOAL_STATE:
                pol = str(pol) + " G"
            elif [i, j] == START_STATE:
                pol = str(pol) + " S"
            else:
                pol = str(pol) + "  "
            optimal_policy[-1].append(pol)  # append to the last row
    print('Optimal policy is:')
    for row in optimal_policy:
        print(row)
    print('Wind strength for each column:\n{}'.format([str(w) for w in WIND_STRENGTH]))

#######################################################################
if __name__ == '__main__':    
    figure_6_3a()
    figure_6_3b()

#######################################################################