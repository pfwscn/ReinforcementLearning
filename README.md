## Part 1 Tabular Solution Methods
### Chapter 6 Temporal-Difference (TD) Learning (on episodic tasks)
#### Associative search with incomplete knowledge of environment (model-free)

1. random_walk06
* Learns from samples (like MC), but updates can be done within an episode (bootstraps like DP)
* one-step TD: TD(0)
* Batch TD: maximum-likelihood model, certainty-equivalence estimate
2. windy_grid_world
* Sarsa on-policy TD control
3. cliff_walking
* Q-learning off-policy TD control
* Expected Sarsa
4. maximization_bias
* Double learning
* Afterstates

python==3.11 <br>
numpy==1.26.4 <br>
matplotlib==3.8.2 <br>
tqdm==4.66.2 <br>
