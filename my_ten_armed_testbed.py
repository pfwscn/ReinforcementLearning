#######################################################################
# Copyright (C)                                                       #
# 2024 SK (github.com/pfwscn)                                         #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Artem Oboturov(oboturov@gmail.com)                             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

#######################################################################
# Chp 2 Multi-armed Bandits

# q*(a) true value of taking action a
# Q_t(a) estimated value of action a at time t
# (2.1) average reward per action taken prior to t

# Action selection methods: NG, UCB
# Greedy: choose action with max estimated value A_t = argmax_a Q_t(a) (2.2)
# Near (epsilon) greedy (NG): greedy with prob. (1-epsilon), random over all actions (epsilon)
# Larger epsilon encourages more exploitation (noisy, non-stationary environments)
# Upper Confidence Bound (UCB): select actions in a preferential way
# (2.10) action-value estimates adjusted by a parameterized (c) square-root term, 
# that is a ratio of time and action counts: ln(t)/N_t(a)
# Exploratory aspect: ratio increases with lower action count, c value
# UCB usually not a practical action selection method in more realistic environments

# Initial values Q_1
# Introduce bias or prior knowledge, useful for value estimation, e.g. encourage early exploration
# "...any method that focuses on initial conditions in any special way is unlikely to help with
# the general nonstationary case."

# Value estimation methods: SA, CSS
# Q_n is estimate value of an action after it has been taken n-1 times
# Q_1 is initial (pre-action) value when an action has been taken 0 times
# (2.1) (SA) sample-average method Q_n+1 = Q_n + 1/n[R_n - Q_n] (2.3)
# Update rule: NewEstimate <- OldEstimate + StepSize [Target - OldEstimate] (2.4)
#   stepsize parameter denoted alpha in (0, 1.0]
# (2.5) Q_n+1 = Q_n + alpha[R_n - Q_n]
# (2.6) Biased Constant Stepsize CSS (exponential recency-weighted average):
#   Q_n+1 is a weighted average of past rewards and the initial estimate Q_1
#   Bias of Q_1 and ealier rewards decays exponentially as n increases
# (2.7) Conditions required to assure convergence with prob. 1 of estimates to true q-values
#   Seldom used in practise
# (2.8) (2.9) Unbiased Constant Stepsize 

# Gradient bandit algorithms: a different approach
# Action selection not reward based (value estimation Q_t(a)), but 
# probability based (phi_t(a) soft-max distribution)
# (2.11) phi_t(a) = P(A_t=a) is probability of taking action a at time t, and
# is a function of preference values (H_t(a)) of all actions in A
# (2.12) Stochastic gradient ascent idea to learn action preferences H_t(a)
# Selecting an action adjusts its H_t value 
# (up or down relative to a baseline (average reward so far)), and
# adjusts the H_t values of all other actions in the opposite direction.

# Section 2.9 
# " ... [in] nonassociative tasks [all tasks considered so far] ... 
# the learner either tries to find a single best action when the task [reward] is stationary,
# or tries to track the best action as it changes over time when the task is nonstationary."  
# "... an associative search task [contextual/multi-situational problem] ... 
# involves both trial-an-error learning to search for the best actions, and
# association of these actions with the situations in which they are best."
# Chp3: evaluate q(s, a) or v(s), not just q(a)
# "If actions are allowed to affect the next situation as well as the reward,
# then we have the full reinforcement learning problem."

# Section 2.10
# Bayesian approach to balance exploration-exploitation tension

#######################################################################

import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
from tqdm import trange
matplotlib.use('Agg')

rng = np.random.default_rng()

def multi_argmax(values):
    max_val = np.max(values)
    return rng.choice(np.where(max_val == values)[0])


class ArmedBandit:
    def __init__(self, bandit_params=dict()):
        self.params = copy.deepcopy(bandit_params)
        self.num_arms = self.params["num_arms"]
        self.arms = np.arange(self.num_arms)  # gradient
        self.environment = self.params["environment"]
        if self.environment == "stationary":
            self.true_values = rng.standard_normal(self.num_arms)
        elif self.environment == "non-stationary":  # Exercise 2.5            
            self.true_values = np.asarray([rng.random()] * self.num_arms)
        elif self.environment == "gradient":    # Section 2.8 Fig. 2.5
            self.true_values = rng.standard_normal(self.num_arms) + 4.0 
        self.best_action = np.argmax(self.true_values)
        self.action_selection = self.params["action_selection"]
        self.value_estimation = self.params["value_estimation"]        
        self.initial_values = self.params["initial_values"]
        self.estimated_values = np.copy(self.initial_values)
        self.action_counts = np.zeros(self.num_arms, dtype=int)
        
    def reset(self):
        if self.environment == "stationary":
            self.true_values = rng.standard_normal(self.num_arms)
        elif self.environment == "non-stationary":
            self.true_values = np.asarray([rng.random()] * self.num_arms)        
        elif self.environment == "gradient":
            self.true_values = rng.standard_normal(self.num_arms) + 4.0                
        self.best_action = np.argmax(self.true_values)
        self.estimated_values = np.copy(self.initial_values)     
        self.action_counts = np.zeros(self.num_arms, dtype=int)        
    
    # take action, generate a reward for arm/action a in [0, k-1]
    def get_reward(self, action):
        reward = rng.normal(self.true_values[action], 1)  # mean, stdev
        if self.environment == "non-stationary":
            reward += rng.normal(0, 0.01)  # Exercise 2.5 random walk, add some random noise
        return reward

    def step(self, timestep, baseline_reward=0.0):
        if self.action_selection == "NG":
            action = self.neargreedy_action_selection(epsilon=self.params["epsilon"])
        elif self.action_selection == "UCB":
            action = self.ucb_action_selection(epsilon=self.params["epsilon"], 
                                               time=timestep, c=self.params["ucb"])
        elif self.action_selection == "gradient":
            action = self.gradient_action_selection()            
        reward = self.take_action(action)        
        if  self.value_estimation == "SA":
            self.simple_value_estimation(action, reward)
        elif self.value_estimation == "CSS":
            self.constantstepsize_value_estimation(action, reward, alpha=self.params["alpha"])
        elif self.value_estimation == "gradient":
            self.gradient_preferential_learning(action, reward, alpha=self.params["alpha"], 
                                                baseline_reward=baseline_reward)
        return action, reward

    def take_action(self, action):        
        reward = self.get_reward(action)
        self.action_counts[action] += 1
        return reward

    def neargreedy_action_selection(self, epsilon):
        if rng.random() < epsilon: # non-greedy, select an arm uniformly at random
            action = rng.integers(0, self.num_arms)
        else:
            action = multi_argmax(self.estimated_values)        
        return action

    def ucb_action_selection(self, epsilon, time, c):
        if rng.random() < epsilon: # non-greedy, select an arm using ucb        
            ucb_values = np.copy(self.estimated_values)
            for a in range(self.num_arms):
                n = self.action_counts[a] + 1e-5  # ensure n != 0.0                
                ucb_values[a] += c * np.sqrt(np.log(time+1)/n) # time starts at 0 
            action = multi_argmax(ucb_values)
        else:
            action = multi_argmax(self.estimated_values)
        return action

    # simple average (2.3)
    def simple_value_estimation(self, action, reward):
        q_n = self.estimated_values[action]  # old/current estimate
        n = self.action_counts[action]
        q_n1 = q_n + 1/n * (reward - q_n)  # new estimate
        self.estimated_values[action] = q_n1
        return q_n1
    
    # biased constant step-size (2.5) (2.6)
    # alpha in (0.0, 1.0]
    def constantstepsize_value_estimation(self, action, reward, alpha):
        q_n = self.estimated_values[action]  # old/current estimate        
        q_n1 = q_n + alpha * (reward - q_n)  # new estimate        
        self.estimated_values[action] = q_n1        
        return q_n1
    
    # estimated_values (Q) are now preferential values(H)
    def gradient_action_selection(self):
        exp_prefs = np.exp(self.estimated_values)  # exp(H_t(a)), initially all exp(0)=1
        self.action_probs = exp_prefs / np.sum(exp_prefs)  # phi_s, initially, uniform at random
        action = rng.choice(self.arms, p=self.action_probs)
        return action
    
    # (2.12)
    def gradient_preferential_learning(self, action, reward, alpha, baseline_reward):
        one_hot = np.zeros(self.num_arms)
        one_hot[action] = 1        
        self.estimated_values += alpha * (reward - baseline_reward) * (one_hot - self.action_probs)
        return        


def simulate(scenarios, num_runs, num_timesteps, gradient_baseline=False):
    rewards = np.zeros((len(scenarios), num_runs, num_timesteps))
    best_actions = np.zeros(rewards.shape)
    for i, kab in enumerate(scenarios):
        for r in trange(num_runs):
            kab.reset()            
            for ts in range(num_timesteps):
                if gradient_baseline:
                    baseline_reward = rewards[i, r].mean()
                    action, reward = kab.step(ts, baseline_reward)
                else:
                    action, reward = kab.step(ts)
                rewards[i, r, ts] = reward
                if action == kab.best_action:
                    best_actions[i, r, ts] = 1
    # for each scenario, reward per timestep averaged over runs 
    reward_averages = rewards.mean(axis=1)
    # for each scenario, freq of runs per timestep that took best action
    best_action_averages = best_actions.mean(axis=1) 
    return reward_averages, best_action_averages

     
#######################################################################

# Reward distribution of 10-armed bandit problem
def figure_2_1():
    print("figure 2_1")
    k = 10
    bandit_params = dict()
    bandit_params["num_arms"] = k
    bandit_params["environment"] = "stationary"
    # bandit_params["environment"] = "non-stationary"
    bandit_params["initial_values"] = np.zeros(k)
    bandit_params["action_selection"] = "NG"    
    bandit_params["value_estimation"] = "SA"    
    kab = ArmedBandit(bandit_params.copy())
    D = [] 
    for a in range(0, kab.num_arms):
        D.append([ kab.get_reward(a) for i in range(0, 2000) ])
    A = np.asarray(D).reshape(2000, kab.num_arms)

    plt.violinplot(dataset=A)
    plt.plot(np.arange(10)+1, kab.true_values, '*')    
    plt.xlabel("Arm/Action")
    plt.ylabel("Reward distribution")
    # plt.ylabel("Non-stationary Reward distribution")
    plt.grid(True)
    plt.savefig('images/my_figure_2_1.png')
    # plt.savefig('images/my_figure_2_1a.png')
    plt.close()


# Simple average value estimation, near greedy action selection
# Effect of different epsilon values
def figure_2_2(num_runs=2000, num_timesteps=1000):
    print("figure 2_2")
    k = 10
    bandit_params = dict()
    bandit_params["num_arms"] = k    
    bandit_params["environment"] = "stationary"
    # bandit_params["environment"] = "non-stationary"
    bandit_params["action_selection"] = "NG"    
    bandit_params["value_estimation"] = "SA"
    bandit_params["initial_values"] = np.zeros(k)    
    scenarios = []
    epsilons = [0, 0.01, 0.1]    
    for eps in epsilons:
        bp = copy.deepcopy(bandit_params)
        bp["epsilon"] = eps
        scenarios.append(ArmedBandit(bp))
    reward_averages, best_action_averages = simulate(scenarios, num_runs, num_timesteps)

    plt.figure(figsize=(10, 20))
    plt.subplot(2, 1, 1)
    for eps, ravg in zip(epsilons, reward_averages):
        plt.plot(ravg, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()
    plt.subplot(2, 1, 2)
    for eps, bavg in zip(epsilons, best_action_averages):
        plt.plot(bavg, label='$\epsilon = %.02f$' % (eps))
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()
    plt.savefig('images/my_figure_2_2.png')
    # plt.savefig('images/my_figure_2_2a.png')
    plt.close()


# Constant step size value estimation, near greedy action selection
# Effect of optimistic initial action-value estimates
# Ex 2.6: Initial spike followed by lag, waiting for initial conditions (explorative phase) 
# to decay, so agent can settle into exploitative mode. Less optimistic (Q_1=3), narrower lag.
def figure_2_3(num_runs=2000, num_timesteps=1000):
    print("figure 2_3")
    k = 10
    bandit_params = dict()
    bandit_params["num_arms"] = k
    bandit_params["environment"] = "stationary"
    # bandit_params["environment"] = "non-stationary"
    bandit_params["action_selection"] = "NG"    
    bandit_params["value_estimation"] = "CSS"
    bandit_params["alpha"] = 0.1
    scenarios = []
    # Realistic Q_1=0; neargreedy eps=0.1
    bp = copy.deepcopy(bandit_params)
    bp["initial_values"] = np.zeros(k)
    bp["epsilon"] = 0.1    
    scenarios.append(ArmedBandit(bp))    
    # Optimistic Q_1=5; greedy eps=0
    bp = copy.deepcopy(bandit_params)
    bp["initial_values"] = np.asarray([5.0] * k) # real datatype!
    bp["epsilon"] = 0.0    
    scenarios.append(ArmedBandit(bp)) 
    # Optimistic Q_1=3; greedy eps=0
    bp = copy.deepcopy(bandit_params)
    bp["initial_values"] = np.asarray([3.0] * k)
    bp["epsilon"] = 0.0
    scenarios.append(ArmedBandit(bp))
    _, best_action_averages = simulate(scenarios, num_runs, num_timesteps)

    plt.plot(best_action_averages[0], label='$Q_1 = 0, \epsilon = 0.1$')
    plt.plot(best_action_averages[1], label='$Q_1 = 5, \epsilon = 0.0$')    
    plt.plot(best_action_averages[2], label='$Q_1 = 3, \epsilon = 0.0$')    
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/my_figure_2_3.png')
    # plt.savefig('images/my_figure_2_3a.png')
    plt.close()


# Simple average value estimation, UCB action selection
# Effect of UCB with different parameters, compared with NG action selection    
# Ex 2.8 Like optimistic initial action-values, UCB also exhibit early stage spike
# because low action counts in the beginning yields larger adjustment ratios
# "If N_t(a)=0, then a is considered to be a maximizing action."    
def figure_2_4(num_runs=2000, num_timesteps=1000):
    print("figure 2_4")
    k = 10
    bandit_params = dict()
    bandit_params["num_arms"] = k
    bandit_params["environment"] = "stationary"    
    bandit_params["initial_values"] = np.zeros(k)
    bandit_params["value_estimation"] = "SA"    
    scenarios = []    
    bp = copy.deepcopy(bandit_params)    
    bp["action_selection"] = "UCB"    
    bp["ucb"] = 2.0
    bp["epsilon"] = 1.0
    scenarios.append(ArmedBandit(bp))    
    bp = copy.deepcopy(bandit_params)
    bp["action_selection"] = "NG"
    bp["epsilon"] = 0.1
    scenarios.append(ArmedBandit(bp))
    bp = copy.deepcopy(bandit_params)
    bp["action_selection"] = "UCB"    
    bp["ucb"] = 1.0
    bp["epsilon"] = 0.9    
    scenarios.append(ArmedBandit(bp))        
    reward_averages, _ = simulate(scenarios, num_runs, num_timesteps)

    plt.plot(reward_averages[0], label='greedy UCB $c = 2$')
    plt.plot(reward_averages[1], label='neargreedy $\epsilon = 0.1$')
    plt.plot(reward_averages[2], label='neargreedy UCB $c = 1$ $\epsilon = 0.9$')
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.legend()
    plt.savefig('images/my_figure_2_4.png')
    plt.close()


# Gradient bandit algorithm, probability based action selection
def figure_2_5(num_runs=2000, num_timesteps=1000):
    print("figure 2_5")
    k = 10
    bandit_params = dict()
    bandit_params["num_arms"] = k
    bandit_params["environment"] = "gradient"
    bandit_params["action_selection"] = "gradient"    
    bandit_params["value_estimation"] = "gradient"
    bandit_params["initial_values"] = np.zeros(k)    
    scenarios = []
    bp = copy.deepcopy(bandit_params)    
    bp["alpha"] = 0.1
    scenarios.append(ArmedBandit(bp)) 
    bp = copy.deepcopy(bandit_params)    
    bp["alpha"] = 0.4
    scenarios.append(ArmedBandit(bp)) 
    # baseline=mean
    _, avgbaseline_best_action_averages = simulate(scenarios, num_runs, num_timesteps, True)
    # baseline=0.0
    _, zerobaseline_best_action_averages = simulate(scenarios, num_runs, num_timesteps)
    best_action_averages = np.stack((avgbaseline_best_action_averages[0],
                                     zerobaseline_best_action_averages[0],
                                     avgbaseline_best_action_averages[1],
                                     zerobaseline_best_action_averages[1]))    
    
    labels = [r'$\alpha = 0.1$, with baseline',
              r'$\alpha = 0.1$, without baseline',
              r'$\alpha = 0.4$, with baseline',
              r'$\alpha = 0.4$, without baseline']
    for i in range(len(best_action_averages)):
        plt.plot(best_action_averages[i], label=labels[i])
    plt.xlabel('Steps')
    plt.ylabel('% Optimal action')
    plt.legend()
    plt.savefig('images/my_figure_2_5.png')
    plt.close()


# Section 2.10 A parameter study by comparing learning curves 
# "In assessing a method, we should attend not just to 
# how well it does at its best parameter setting,
# but also to how sensitive it is to its parameter value."    
def figure_2_6(num_runs=2000, num_timesteps=1000):
    print("figure 2_6")
    k = 10
    bandit_params = dict()
    bandit_params["num_arms"] = k     
    bandit_params["environment"] = "stationary"
    
    labels = ['epsilon-greedy', 'optimistic initialization', 'UCB', 'gradient bandit'] 
    parameters = [ np.arange(-7, -1, dtype=float), np.arange(-2, 3, dtype=float), 
                  np.arange(-4, 3, dtype=float), np.arange(-5, 2, dtype=float) ]
        
    # Neargreedy action selection, simple average value estimation    
    scenarios = []
    epsilons = [ pow(2.0, i) for i in parameters[0] ]
    bp = copy.deepcopy(bandit_params)
    for eps in epsilons:        
        bp["action_selection"] = "NG"         
        bp["epsilon"] = eps
        bp["initial_values"] = np.zeros(k)    
        bp["value_estimation"] = "SA"         
        scenarios.append(ArmedBandit(copy.deepcopy(bp)))
    reward_averages,_ = simulate(scenarios, num_runs, num_timesteps)
    method_rewards = np.mean(reward_averages, axis=1)
    print(epsilons)
    print(method_rewards)
    plt.plot(parameters[0], method_rewards, label=labels[0])    
    del bp, scenarios
    
    # Optimistic initialization, greedy action selection, constant step size value estimation
    scenarios = []
    initial_values = [ pow(2.0, i) for i in parameters[1] ]
    bp = copy.deepcopy(bandit_params)
    for v0 in initial_values:        
        bp["action_selection"] = "NG"
        bp["epsilon"] = 0.0  # 100% greedy action selection
        bp["initial_values"] = np.asarray([v0] * k)        
        bp["value_estimation"] = "CSS" 
        bp["alpha"] = 0.1
        scenarios.append(ArmedBandit(copy.deepcopy(bp)))
    reward_averages,_ = simulate(scenarios, num_runs, num_timesteps)
    method_rewards = np.mean(reward_averages, axis=1)
    print(initial_values)
    print(method_rewards)
    plt.plot(parameters[1], method_rewards, label=labels[1])
    del bp, scenarios
    
    # UCB action selection, simple average value estimation
    scenarios = []
    ucbs = [ pow(2.0, i) for i in parameters[2] ]
    bp = copy.deepcopy(bandit_params)
    for c in ucbs:
        bp["action_selection"] = "UCB"
        bp["ucb"] = c
        bp["epsilon"] = 1.0  # 100% UCB action selection
        bp["initial_values"] = np.zeros(k)    
        bp["value_estimation"] = "SA"         
        scenarios.append(ArmedBandit(copy.deepcopy(bp)))
    reward_averages,_ = simulate(scenarios, num_runs, num_timesteps)
    method_rewards = np.mean(reward_averages, axis=1)
    print(ucbs)
    print(method_rewards)
    plt.plot(parameters[2], method_rewards, label=labels[2])    
    del bp, scenarios
    
    # Gradient bandit, with baseline
    scenarios = []
    alphas = [ pow(2.0, i) for i in parameters[3] ]
    bp = copy.deepcopy(bandit_params)
    for a in alphas:
        # bp["environment"] = "gradient"
        bp["action_selection"] = "gradient"
        bp["initial_values"] = np.zeros(k)    
        bp["value_estimation"] = "gradient"
        bp["alpha"] = a
        scenarios.append(ArmedBandit(copy.deepcopy(bp)))
    reward_averages,_ = simulate(scenarios, num_runs, num_timesteps, True)
    method_rewards = np.mean(reward_averages, axis=1)
    print(alphas)
    print(method_rewards)    
    plt.plot(parameters[3], method_rewards, label=labels[3])
    del bp, scenarios
    
    plt.xlabel('Parameter($2^x$)')
    plt.ylabel('Average reward')
    plt.legend()
    plt.savefig('images/my_figure_2_6.png')
    plt.close()


if __name__ == '__main__':
    figure_2_1()
    figure_2_2()
    figure_2_3()
    figure_2_4()
    figure_2_5()
    figure_2_6()

#######################################################################