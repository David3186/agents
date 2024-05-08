# agents
CSCI 381 Final Project Spring 2024 
Maxim Enis and David Geotze

Info:
`agents.py`: Contains main code for training and playing agents. To run an agent, construct an `Agent`, which takes a `game_type` (kwargs to specify in a `gymnasium` environment) and a policy network (input must match environment observation dimensions, plus a batch dimension, plus a sequential frame dimension; output must match action size dimension). To specify a pretrained agent, load a pretrained policy network. Otherwise, train the agent using `agent.train` with a specified `TrainConfig`. An example pipeline is in specified in the `__main__` block in `agent.py`. 

As the policy network trains, the reward will be specified in the `running_avg.png` file under the `models/network_dir` directory. The orange line is the baseline and the blue is the running average in reward of the past 50 iterations.

We provide two example pretrained policy networks: one for `CartPole-v1`, and one for `ALE/SpaceInvaders-v5`. We observe that the CartPole network fully solves the game, and `ALE/SpaceInvaders-v5` beats the baseline.



