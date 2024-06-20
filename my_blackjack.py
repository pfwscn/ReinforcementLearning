#######################################################################
# Copyright (C)                                                       #
# 2024 SK (github.com/pfwscn)                                         #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# 2017 Nicky van Foreest(vanforeest@gmail.com)                        #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

#######################################################################
# Chp 5 Monte Carlo (MC) methods (on episodic tasks)
# MC methods require only experience - sample sequences of states, actions
# and rewards, from actual or simulated interaction with an environment,
# not complete knowledge of environment dynamics p(s', r| s, a) (3.2).
# Deals with sample transition probabilities, and averaging sample returns. 
#
# Value estimates and policies updated on completion of an episode only. 
# An episode: S_0(G_0), A_0, R_1, S_1(G_1), A_1, R_2, S_2(G_2), ...,
#                           S_T-1(G_T-1), A_T-1, R_T, S_T(G_T=0) (3.1)
# "Because all action selections are undergoing learning, the problem becomes
# non-stationary, from the point of view of the earlier state."
#
# Policy Evaluation (prediction)
# Section 5.1 MC estimation of state-values 
# v_pi(s) = E_pi[G_t|S_t=s] (3.12) 
# Value of a state S_t=s is expected return G_t = R_t+1 + gamma G_t+1 (3.9), 
#   expected cumulative future discounted reward, starting from s.
# Estimate v_pi(s), the value of a state under policy pi, given 
# a set? of episodes obtained by following pi, that pass thru (visit) s.
#   *first-visit MC average returns from first visits to s only
#   every-visit MC average returns from all visits to s (chps 9, 12)
# Backup diagram for MC estimation of v_pi, begins at a state node,
#   traces the complete trajectory of an episode, and ends at a terminal
#   state node, i.e. S_0, A_0, S_1, A_1,...,S_T; shows only sampled 
#   transitions in the episode
# State values are independent of each other (unlike in DP); 
#   MC does not bootstrap, i.e. build estimates from estimates (Bellman equation).
# Can evaluate value of a single state without evaluating any other states.
# Cost of estimating the value of a single state is independent of the
#   number of states...can generate episodes to target valuation of states
#   of interest...makes MC method more efficient than iterative method (DP).
#
# Section 5.2 MC estimation of action-values
# Absent a model, estimating action-values is more useful than est. state-values.
# A primary goal of MC is to estimate q*(s, a), value of taking action a at
#   state s, under the optimal policy. 
# q_pi(s, a) = E_pi[G_t|S_t=s, A_t=a] (3.13)
# Value of taking action a at state s is the expected return from (s, a), given
# episodes obtained by following pi, where (s, a) occurs or is visited.
#   *first-visit MC average returns from first visits to (s, a) only
#   every-visit MC average returns from all visits to (s, a)
# Complication: many state-action pairs may never be visited, 
#   e.g. deterministic policy, rare (s, a); 
#   problem is no alternatives to compare with (optimal relative to what?).
# General problem of maintaining exploration: "For policy evaluation
#   to work for action values, we must assure continual exploration."
# Assumption of exploring starts (ES): start episodes in a state-action pair,
#   and every (s, a) pair has nonzero probability to be start of an episode.
#   Not realistic, alternative is to only consider stochastic policies.
#
# On vs. off-policy methods (Section 5.4)
# "On-policy methods attempt to evaluate or improve the policy that is used 
# to make decisions, whereas off-policy methods evaluate or improve a policy 
# different from that used to generate the data."
#
# On-policy methods generally use soft policies, i.e. pi(a|s) > 0 for all states 
# (all actions possible from a state have non-zero probability), and gradually 
# shift to a deterministic optimal policy.
# epsilon-greedy policies are examples of epsilon-soft policies, with epsilon > 0
#
# Off-policy methods (Section 5.5) involve two policies: 
# one that is learned (target, pi), and one that explores (behavior, b).
# off-policy learning: "learning is from data that is off the target policy"
# more powerful and general, but of greater variance and slower to converge
#
# Assumption of coverage: pi(a|s) > 0 implies b(a|s) > 0
#   target(pi) deterministic, greedy/optimizing; behavior(b) stochastic, exploratory
#
# Uses importance-sampling: a general technique to estimate expected values under
#   one distribution given samples from another.
# Importance-sampling ratio (rho) is the relative probability of a trajectory 
#   under the target and behavior policies (5.3)
# Expected returns under pi, can be estimated from returns generated under b, as
#   v_pi(s) = E[rho_t:T-1 G_t | S_t=s] (5.4)
# Predict V_pi(s) from a bunch of episodes sampled under b:
#   Ordinary importance-sampling: simple average of rho-adjusted returns (5.5)
#   Weighted importance-sampling: weighted average of rho-adjusted returns (5.6)
# Variances, biases of estimates: first or every visit, ordinary or weighted average
#
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

rng = np.random.default_rng()

# Example 5.1 Blackjack
# Goal: obtain cards whose sum is as large as possible without exceeding 21
# Cards: numerical 2...9, face 10, ace 1 or 11
# Independent player(s) vs. dealer
# Game begins: player and dealer (one up, one down) receive two cards each 
# (1) Either one or both parties are natural (10 + ace = 21); outcome win or draw, game ends.
# (2) Player request (hits) one card at a time, until either player stops (sticks), 
# or hand (sum of cards) exceeds 21 (bust). 
# If sticks, dealer's turn; if bust, player loses (dealer wins), game ends.
# (3) Dealer strategy/fixed policy: hits while less than 17, and sticks otherwise. 
# If dealer's sum of cards exceed 21, dealer goes bust and loses (player wins), game ends.
# Otherwise (both player and dealer sticks), the party with the larger hand wins.

# Episodic finite MDP. Each game is an episode.
# Rewards: +1, -1, 0 for win, lose and draw respectively, collected upon reaching terminal state. 
# Zero rewards for all other actions. 
# No discounting (gamma=1); so terminal rewards are also returns.
# Actions: hit, stick
# States: player's cards (hand), and dealer's showing card
# Cards are dealt with replacement (from an infinite deck).
# An ace is usable, if it can be counted as 11 without exceeding 21.
# If sum <= 11, player should always hit (max is 10); decisions only necessary when sum in [12 ... 21]
# Total of 200 states: 10 * 10 * 2; current sum [12...21], dealer's card [ace...10], usable ace [y/n]

# Renamed stand to stick and revalued hit=1, stick=0
STICK = 0; HIT = 1
ACTIONS = [STICK, HIT]

# Player is the learning agent, so target and behavior policies pertain to player
# Policy for player: player hits up to 19, sticks on 20 and 21
PLAYER_POLICY = np.ones(22, dtype=int) # 0...21
PLAYER_POLICY[20] = STICK
PLAYER_POLICY[21] = STICK

# function form of target policy of player
# state = (player_total, player_usable_ace, dealer_card1)
def target_policy(state):
    return PLAYER_POLICY[state[0]]  # state[0] is player_total

# function form of behavior policy of player
# effectively a Bernoulli trial, equiprob HIT or STICK, ignore state info
def random_behavior_policy(state):
    return rng.choice(ACTIONS, shuffle=False)
    # if rng.integers(0, 1, endpoint=True):
    #     return HIT
    # return STICK

# Dealer's policy is fixed: hits up to 16, and sticks thereafter 17...21
DEALER_POLICY = np.ones(22, dtype=int) # 0...21
for i in range(17, 22):
    DEALER_POLICY[i] = STICK

# Reward to player is obtained only when game ends (episode terminates)
WINS = 1
TIES = 0
LOSES = -1


# Deal card, hit action
def get_card():    
    card = rng.integers(1, 14)  # 1...13
    if 1 == card: # ace
        return 11
    elif card > 9: # 10, face (J, Q, K)
        return 10
    else:
        return card  # 1...9

# An usable ace is one that contributes 11 to the total, without going bust!
# returns an integer and a boolean    
def sum_cards(cards: list):
    total = sum(cards, start=0)
    aces = np.where(11 == np.array(cards))[0].tolist() # indices of the ace cards
    # reduce total if necessary by converting usable aces (11) to contribute 1 to total
    while total > 21 and len(aces) > 0:  
        total = total - 10  # -11 + 1
        assert 11 == cards[aces[-1]]
        cards[aces.pop()] = 1
    usable_ace = (total < 22 and len(aces) > 0)
    return total, usable_ace

# Simulate a blackjack game, generates a player episode
# Player is agent learning the game: rewarded +1 if wins, -1 if losses and 0 if ties
# player_policy: (specifies) player policy (function)
# game_state: player sum, player usable ace, dealer's one (showing) card
# start_action: the player's action from given game_state
def play(player_policy, game_state=None, start_action=None):      
    dealer_card1 = None; dealer_cards=[]; dealer_action = None
    dealer_total = 0; dealer_usable_ace = False
    player_cards = []; player_action = None
    player_total = 0; player_usable_ace = False    
    state = None; player_episode = []  # player's sequence of (s, a) tuples in a game
    
    if game_state is None:  # start fresh game; dealer and player receives two cards each
        dealer_card1 = get_card()
        dealer_cards.append(dealer_card1); dealer_cards.append(get_card())
        dealer_total, dealer_usable_ace = sum_cards(dealer_cards)
        player_cards.append(get_card()); player_cards.append(get_card())
        player_total, player_usable_ace = sum_cards(player_cards)
    else: # use given game state
        player_total, player_cards, player_usable_ace, dealer_card1 = game_state
        dealer_cards.append(dealer_card1); dealer_cards.append(get_card()); 
        dealer_total, dealer_usable_ace = sum_cards(dealer_cards)
    
    # current state of the game, don't need player_cards
    # state = game_state.copy()    
    # state = (player_total, player_cards.copy(), player_usable_ace, dealer_card1)
    state = (player_total, player_usable_ace, dealer_card1)

    # game ends if either dealer or player or both get 21
    # return the last/terminal state, reward, and trajectory
    # generally don't include the last/terminal state in an episode since there's no further action
    # but the agent must also be aware of the natural state which terminates a game without
    # any action taken, so need to include them in episodes (made a difference for Fig 5.2!)
    if 21 == dealer_total:
        player_reward = (TIES if 21 == player_total else LOSES)
        player_episode.append((state, STICK)) 
        return state, player_reward, player_episode
    elif 21 == player_total:
        player_reward = WINS
        player_episode.append((state, STICK))
        return state, player_reward, player_episode
        
    assert dealer_total < 21 and player_total < 21
    # game starts with player's turn
    player_action = (player_policy(state) if start_action is None else start_action)        
    while True:
        # track player's trajectory for importance sampling        
        player_episode.append((state, player_action)) 
        if player_action == STICK: 
            break  # game has not ended, no reward yet
        # action is hit
        player_cards.append(get_card())  
        player_total, player_usable_ace = sum_cards(player_cards)
        state = (player_total, player_usable_ace, dealer_card1)
        if player_total > 21:
            return state, LOSES, player_episode  # on 21 player may tie if dealer also gets 21            
        player_action = player_policy(state) 
                  
    # dealer's turn, doesn't affect player's state or player episode
    # affects player reward, game outcome
    while True:        
        dealer_action = DEALER_POLICY[dealer_total]
        if dealer_action == STICK:
            break
        dealer_cards.append(get_card())  # action is hit
        dealer_total, dealer_usable_ace = sum_cards(dealer_cards)        
        if dealer_total > 21:  # player wins            
            return state, WINS, player_episode

    # both player and dealer stick 
    assert player_action == STICK and dealer_action == STICK
    assert player_total < 22 and dealer_total < 22
    if player_total > dealer_total:
        return state, WINS, player_episode
    elif player_total == dealer_total:
        return state, TIES, player_episode
    else:
        return state, LOSES, player_episode

#######################################################################
# adjust for indexing, doesn't convert player_usable_ace from bool to int!
def process_state(state):
    player_total, player_usable_ace, dealer_card1 = state
    # player total range 12...21; -12 so that indices are 0...9
    player_total = player_total - 12  
    # card value range 11, 1...10; -11 or -1 so that indices are 0...9
    if 11 == dealer_card1:
        dealer_card1 = dealer_card1 - 11
    else:
        dealer_card1 = dealer_card1 - 1        
    return player_total, player_usable_ace, dealer_card1

#######################################################################
# Monte Carlo (On-Policy) Prediction (PE)  (Fig 5.1)
# Addresses the prediction problem of estimating state-values V, 
# or state-value function of each state for a given policy (policy evaluation), 
# in this case the player's target policy.
# On-policy: use a single policy to explore, evaluate and improve.
# 
# State-values converge to their true value as number of state visits goes to infinity.
# State-value function is expected return at each observed state, for t < T-1: 
#   G_t = r(S_t, A_t, S_t+1) + gamma G_t+1 
#       where r(S_t, A_t, S_t+1) = 0 since there is no reward for entering a non-terminal state    
#   G_T-1 = r(S_T-1, A_T-1, S_T) in {+1, 0, -1} + gamma G_T where gamma=1.0 and G_T=0.
# Essentially all states in an episode gets the final reward as their return (value in this episode).     
# Applies (simple) arithmetic average, to estimate expected value of a state, over all episodes.  
# States are identified by three factors: player total, player usable ace and dealer's showing card;
#   player cards are 'hidden' (redundant).  
# Total of 200 states: 10 * 2 * 10; player's total [12...21] when a HIT/STICK action decision is necessary, 
#   player's usable ace [y/n], dealer's revealed card [ace...10]
#  
# visit_type is First or Every (default)    
def monte_carlo_on_policy(num_episodes, visit_type='Every'):
    # arrays to calculate state-values; similar to armed-bandit return function
    # keep track of number of visits and accumulated returns at each state
    # ua = usable ace, nua = no usable ace
    ua_state_returns = np.zeros((10, 10))   # accumulate state returns 
    ua_state_counts = np.ones((10, 10))     # count state visits
    nua_state_returns = np.zeros((10, 10))        
    nua_state_counts = np.ones((10, 10))    # initialize 1 to avoid division by 0
    episode_lens = []
    
    # Fig. 5.3
    player_total = 13
    player_usable_ace = True
    dealer_card1 = 2
    
    for i in tqdm(range(0, num_episodes)):
        terminal_state, reward, player_episode = play(target_policy)  # start game fresh
        
        # Fig. 5.3
        # player_cards = get_cards(13, True) 
        # game_state = (13, player_cards, True, 2)        
        # terminal_state, reward, player_episode = play(target_policy, game_state)
        
        episode_lens.append(len(player_episode))
        
        states_visited = list()
        for (state, action) in player_episode:            
            player_total, player_usable_ace, dealer_card1 = process_state(state)
            
            if visit_type == "Every":
                if player_usable_ace:
                    ua_state_returns[player_total, dealer_card1] += reward
                    ua_state_counts[player_total, dealer_card1] += 1                
                else:
                    nua_state_returns[player_total, dealer_card1] += reward
                    nua_state_counts[player_total, dealer_card1] += 1
            elif visit_type == "First" and state not in states_visited:
                states_visited.append(state)
                if player_usable_ace:
                    ua_state_returns[player_total, dealer_card1] += reward
                    ua_state_counts[player_total, dealer_card1] += 1                
                else:
                    nua_state_returns[player_total, dealer_card1] += reward
                    nua_state_counts[player_total, dealer_card1] += 1
            else:
                print("state already visited")    
        
        # Fig. 5.3
        # 1 (-0.278,930), 10 (-0.277,844), 50 (-0.277,271), 80 (-0.277,207), 100 million
        # if i in (1000000, 3000000, 10000000, 50000000, 80000000, 100000000): 
        #     ua_state_values = ua_state_returns/ua_state_counts
        #     true_value = ua_state_values[13-12, 2-1]
        #     print(i, true_value)
    
    # Exercise 5.2 little difference expected between first and every-visit MC
    # Player_total behavior increases monotonically, except with usable aces, 
    # state revists are possible though rare, 
    # e.g. player in state 12 with cards (11, 11), HITs, and obtains hand (11, 11, 10),
    # which puts him back in state 12.
    # Episodes tend to be short (right-skewed); sample stats: average 1.8 var 0.8 range 1..7.
    print("Episode lengths statistics (min, max, mean, var, bincount)", 
          np.min(episode_lens), np.max(episode_lens), 
          np.mean(episode_lens), np.var(episode_lens), np.bincount(episode_lens))
    # Estimated state-values
    return ua_state_returns/ua_state_counts, nua_state_returns/nua_state_counts


# Exercise 5.1: policies are player sticks on 20+, and dealer sticks on 17+;
# when a player's total is 20 or 21, a win is more likely (higher state values)
# A usable ace helps player to avoid bust (losing), so those states tend to have higher value.
# A usable ace also helps dealer avoid bust, so those states (for player) tend to have lower value.
#
# MC estimates of state-value functions after 10,000 and afer 500,000 episodes (games)
# visit_type is First or Every (default)
def figure_5_1(visit_type='Every'):
    ua_state_values_10K, nua_state_values_10K = monte_carlo_on_policy(10000, visit_type)
    ua_state_values_500K, nua_state_values_500K = monte_carlo_on_policy(500000, visit_type)

    state_values = [ ua_state_values_10K, ua_state_values_500K,
                    nua_state_values_10K, nua_state_values_500K ]
    titles = [ 'Usable Ace, 10000 Episodes', 'Usable Ace, 500000 Episodes',
              'No Usable Ace, 10000 Episodes', 'No Usable Ace, 500000 Episodes' ]
    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()
    for svalue, title, axis in zip(state_values, titles, axes):
        fig = sns.heatmap(np.flipud(svalue), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)
    plt.savefig('images/my_figure_5_1_' + visit_type + '.png')
    plt.close()


#######################################################################
# Section 5.3 MC Control (Policy Iteration)
# "... alternates between evaluation and improvement on an episode-by-episode basis.
# After each episode, the observed returns are used for policy evaluation, and then
# the policy is improved at all states visited in the episode."
# "... all returns for each state-action pair are accumulated and averaged, irrespective       
# of what policy was in force when they were observed."
#    
# Monte Carlo with Exploring Starts (Fig. 5.2)
# This is also an on-policy method, but with policy improvement, not just evaluation as above.
# To ensure agent is learning the best possible action from each state, 
#   all feasible actions from each state need to have a non-zero probability.
# Exploring starts addresses the problem of adequate exploration for learning an optimal policy.
# Example 5.3

def get_cards(card_total, usable_ace):
    cards = []
    balance = card_total
    if usable_ace:
        cards.append(11)
        balance = card_total - 11
    while balance > 1:
        c = get_card()  # 2...11
        if c <= balance:
            cards.append(c)
            balance = balance - c       
    if 1 == balance:
        cards.append(1)
    rng.shuffle(cards)  # not necessary
    assert sum(cards, start=0) == card_total    
    return cards.copy()

# argmax with ties broken arbitrarily
def multi_argmax(values):    
    max_val = np.max(values)    
    max_args = np.where(np.abs(values - max_val) < 1e-9)[0].tolist()  # comparing reals    
    max_arg = rng.choice(max_args, shuffle=False)
    return max_arg, max_val, max_args

# Exploring starts (ES)
def monte_carlo_es(num_episodes):
    # 4d array: state=(player_total, player_usable_ace, dealer_card1), player_action
    # Accumulate returns for taking an action in a state (s, a), and following the policy thereafter
    state_action_returns = np.zeros((10, 2, 10, 2))
    # initialze counts to 1 to avoid division by 0
    state_action_counts = np.ones((10, 2, 10, 2))
    
    def greedy_behavior_policy(state):
        player_total, player_usable_ace, dealer_card1 = process_state(state) 
        # explicit conversion from bool to int necessary for indexing!
        player_usable_ace = int(player_usable_ace)          
        # compute expected return for actions [stick, hit] in a state
        action_values = state_action_returns[player_total, player_usable_ace, dealer_card1, :] / \
                  state_action_counts[player_total, player_usable_ace, dealer_card1, :]                
        # choose action with max expected return, break ties arbitrarily
        max_arg, max_val, max_args = multi_argmax(action_values)
        # action = rng.choice([a for a, v in enumerate(action_values) if v == np.max(action_values)],
        #                     shuffle=False)
        # if np.max(action_values) > 0.0:
        #     print(action_values)
        #     print([a for a, v in enumerate(action_values) if v == np.max(action_values)])
        #     print(action)
        # return action
        return max_arg

    # play for several episodes
    for i in tqdm(range(0, num_episodes)):    
        # for each episode, use a randomly initialized state and action
        player_total = rng.integers(12, 22)  # max is 21
        # player_total = np.random.choice(range(12, 22))
        player_usable_ace = bool(rng.integers(0, 1, endpoint=True))
        # player_usable_ace = bool(np.random.choice([0, 1]))
        player_cards = get_cards(player_total, player_usable_ace)
        dealer_card1 = get_card()
        game_state = (player_total, player_cards, player_usable_ace, dealer_card1)
        start_action = rng.choice(ACTIONS, shuffle=False)
        # start_action = np.random.choice(ACTIONS)
        # initial player policy is target_policy
        player_policy = greedy_behavior_policy if i > 0 else target_policy
        terminal_state, reward, player_episode = play(player_policy, game_state, start_action)
        visited_state_action = list() 
        for (state, action) in player_episode:
            player_total, player_usable_ace, dealer_card1 = process_state(state)
            player_usable_ace = int(player_usable_ace)            
            state_action = (player_total, player_usable_ace, dealer_card1, action)
            if state_action in visited_state_action:  
                continue  # if this (s, a) already visited in this episode, go to next (s, a)
            visited_state_action.append(state_action)
            # update values of state-action pairs
            state_action_returns[state_action] += reward
            state_action_counts[state_action] += 1
            # state_action_returns[player_total, player_usable_ace, dealer_card1, action] += reward
            # state_action_counts[player_total, player_usable_ace, dealer_card1, action] += 1
            
    return state_action_returns / state_action_counts


# q(s, a) not testing convergence of values, just specify number of episodes
def figure_5_2():
    # 10, 2, 10, 2
    state_action_values = monte_carlo_es(500000) 
    # ua = usable ace; nua = no usable ace; max over the last axis
    ua_state_values = np.max(state_action_values[:, 1, :, :], axis=-1)
    nua_state_values = np.max(state_action_values[:, 0, :, :], axis=-1)
    
    # get the optimal policy: Stick = 0, Hit = 1
    ua_state_actions = np.argmax(state_action_values[:, 1, :, :], axis=-1)
    nua_state_actions = np.argmax(state_action_values[:, 0, :, :], axis=-1)       
    # ua_state_actions = np.zeros((10, 10))
    # nua_state_actions = np.zeros((10, 10))    
    # for i in range(10):  
    #     for j in range(10):  
    #         # ua_states
    #         max_val = ua_state_values[i, j]
    #         action_vals = state_action_values[i, 1, j, :]
    #         ua_state_actions[i, j] = np.random.choice(np.where(max_val == action_vals)[0].tolist())
    #         # nua_states
    #         max_val = nua_state_values[i, j]
    #         action_vals = state_action_values[i, 0, j, :]
    #         nua_state_actions[i, j] = np.random.choice(np.where(max_val == action_vals)[0].tolist())                
    
    images = [ua_state_actions,
              ua_state_values,
              nua_state_actions,
              nua_state_values]

    titles = ['Optimal policy with usable Ace',
              'Optimal value with usable Ace',
              'Optimal policy without usable Ace',
              'Optimal value without usable Ace']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for image, title, axis in zip(images, titles, axes):
        fig = sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)

    plt.savefig('images/my_figure_5_2.png')     
    plt.close()


#######################################################################
# Monte Carlo without Exploring Starts
# As above, also on-policy and first-visit, but epsilon-soft policy not max greedy
#    
def monte_carlo_esoft(num_episodes, epsilon=0.05):
    state_action_returns = np.zeros((10, 2, 10, 2))
    state_action_counts = np.ones((10, 2, 10, 2))
    
    def esoft_behavior_policy(state):
        if rng.random() < epsilon:
            return rng.choice(ACTIONS, shuffle=False)
            # return np.random.choice(ACTIONS)
        player_total, player_usable_ace, dealer_card1 = process_state(state)
        player_usable_ace = int(player_usable_ace)
        action_values = state_action_returns[player_total, player_usable_ace, dealer_card1, :] / \
                  state_action_counts[player_total, player_usable_ace, dealer_card1, :]                
        # choose action with max expected return, break ties arbitrarily
        max_arg, max_val, max_args = multi_argmax(action_values)
        return max_arg
        
    # play for several episodes
    for i in tqdm(range(0, num_episodes)): 
        player_policy = esoft_behavior_policy # if i > 0 else target_policy
        terminal_state, reward, player_episode = play(player_policy)
        visited_state_action = list()         
        for (state, action) in player_episode:
            player_total, player_usable_ace, dealer_card1 = process_state(state)
            player_usable_ace = int(player_usable_ace)            
            state_action = (player_total, player_usable_ace, dealer_card1, action)
            if state_action in visited_state_action:  
                continue  # if this (s, a) already visited in this episode, go to next (s, a)
            visited_state_action.append(state_action)
            # update values of state-action pairs
            state_action_returns[state_action] += reward
            state_action_counts[state_action] += 1
    return state_action_returns / state_action_counts


def figure_5_2a():
    epsilon = 0.05 # larger more exploratory, less greedy
    state_action_values = monte_carlo_esoft(1000000, epsilon) 
    # ua = usable ace; nua = no usable ace
    ua_state_values = np.max(state_action_values[:, 1, :, :], axis=-1)
    nua_state_values = np.max(state_action_values[:, 0, :, :], axis=-1)
    
    # get the optimal policy me: Stick = 0, Hit = 1
    ua_state_actions = np.argmax(state_action_values[:, 1, :, :], axis=-1)
    nua_state_actions = np.argmax(state_action_values[:, 0, :, :], axis=-1)    
    
    images = [ua_state_actions,
              ua_state_values,
              nua_state_actions,
              nua_state_values]

    titles = ['Optimal policy with usable Ace',
              'Optimal value with usable Ace',
              'Optimal policy without usable Ace',
              'Optimal value without usable Ace']

    _, axes = plt.subplots(2, 2, figsize=(40, 30))
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    axes = axes.flatten()

    for image, title, axis in zip(images, titles, axes):
        fig = sns.heatmap(np.flipud(image), cmap="YlGnBu", ax=axis, xticklabels=range(1, 11),
                          yticklabels=list(reversed(range(12, 22))))
        fig.set_ylabel('player sum', fontsize=30)
        fig.set_xlabel('dealer showing', fontsize=30)
        fig.set_title(title, fontsize=30)
    
    plt.savefig('images/my_figure_5_2_esoft_' + str(epsilon) + '.png')
    # plt.savefig('images/my_figure_5_2_hybrid.png')  # first half esoft policy, then greedy policy      
    plt.close()


#######################################################################
# Monte Carlo Sample with Off-Policy
# Example 5.4
#    
def monte_carlo_off_policy(num_episodes):
    player_total = 13
    player_usable_ace = True
    dealer_card1 = 2

    rhos = []  # importance-sampling ratio for each episode
    returns = []  # G
    for i in range(num_episodes):
        player_cards = get_cards(player_total, player_usable_ace) 
        game_state = (player_total, player_cards, player_usable_ace, dealer_card1)        
        terminal_state, reward, player_episode = play(random_behavior_policy, game_state)        
        # only looking at estimating the value of one state! [13, True, 2]
        returns.append(reward)  
        numerator = 1.0; denominator = 1.0
        for (state, action) in player_episode:
            if action == target_policy(state):  # the action is in/on target policy
                # target policy is deterministic pi(a|s)=1.0; behavior policy is b(a|s)=0.5
                # so just adjust denominator
                denominator = denominator * 0.5  
            else:
                # the action is out/off target policy
                numerator = 0.0  
                break
        rho = numerator / denominator # (5.3)
        rhos.append(rho)
        
    rhos = np.asarray(rhos)
    returns = np.asarray(returns)
    adjusted_returns = rhos * returns  # (5.4) expected value of state [13, True, 2] per episode

    adjusted_returns = np.add.accumulate(adjusted_returns) # cumsum
    
    # expected state-value with ordinary importance sampling (5.5)
    ordinary_sampling = adjusted_returns / np.arange(1, num_episodes + 1)  
    
    # expected state-value with weighted importance sampling (5.6)
    rhos = np.add.accumulate(rhos)
    with np.errstate(divide='ignore', invalid='ignore'):
        weighted_sampling = np.where(rhos != 0, adjusted_returns / rhos, 0)

    return ordinary_sampling, weighted_sampling


# player_total = 13
# player_usable_ace = True
# dealer_card1 = 2
def figure_5_3():
    # true_value = -0.27726  # check this figure with my blackjack implementation
    # 80 million episodes every-visit MC on-(target-)policy game_state=[13, True, 2]
    true_value = -0.277200  
    num_episodes = 10000
    runs = 100  # batches of episodes
    error_ordinary = np.zeros(num_episodes)  # accumulate squared error over runs, then average
    error_weighted = np.zeros(num_episodes)
    for i in tqdm(range(0, runs)):
        ordinary_sampling, weighted_sampling = monte_carlo_off_policy(num_episodes)
        # get the squared error
        error_ordinary += np.power(ordinary_sampling - true_value, 2)
        error_weighted += np.power(weighted_sampling - true_value, 2)
    error_ordinary /= runs
    error_weighted /= runs

    plt.plot(np.arange(1, num_episodes + 1), error_ordinary, color='green', label='Ordinary Importance Sampling')
    plt.plot(np.arange(1, num_episodes + 1), error_weighted, color='red', label='Weighted Importance Sampling')
    plt.ylim(-0.1, 5)
    plt.xlabel('Episodes (log scale)')
    plt.ylabel(f'Mean square error\n(average over {runs} runs)')
    plt.xscale('log')
    plt.legend()

    plt.savefig('images/my_figure_5_3.png')
    plt.close()

#######################################################################
# Visualizations could be improved for more meaningful comparisons,
# e.g. connect/standardize scaling of heatmaps across figures.
#
if __name__ == '__main__':

    # MC on-policy
    # PE policy evaluation only (fixed target policy)    
    figure_5_1(visit_type='First')  
    figure_5_1(visit_type='Every') 
    #
    # PE + PI policy evaluation and improvement; first-visit
    figure_5_2()  # greedy policy, with exploring starts    
    figure_5_2a()  # esoft policy, without exploring starts

    # MC off-policy: target and behavior policy; every-visit
    # Importance sampling: ordinary, weighted
    # Evaluate value of a single state, without evaluating any other states
    #
    # ua_states, _ = monte_carlo_on_policy(100000000) # 100 million 10^8
    # true_value = ua_states[13-12, 2-1]      
    # print(true_value)
    #
    figure_5_3()
    
#######################################################################
    