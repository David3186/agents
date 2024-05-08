# agents

## CSCI 381 Final Project Spring 2024 (Deep Q Learning)

Maxim Enis and David Goetze

### Info:

`agents.py`: Contains main code for training and playing agents. To run an agent, construct an `Agent`, which takes a `game_type` (kwargs to specify in a `gymnasium` environment; `id` is used to specify the name of the environment) and a policy network (input must match environment observation dimensions, plus a batch dimension, plus a sequential frame dimension; output must match action size dimension). To specify a pretrained agent, load a pretrained policy network. Otherwise, train the agent using `agent.train` with a specified `TrainConfig`. An example pipeline is in specified in the `__main__` block in `agent.py`.

As the policy network trains, the reward will be specified in the `running_avg.png` file under the `models/network_dir` directory. The orange line is the baseline and the blue is the running average in reward of the past 50 iterations.
  
We provide two example pretrained policy networks: one for `CartPole-v1`, and one for `ALE/SpaceInvaders-v5`. We observe that the CartPole network fully solves the game, and `ALE/SpaceInvaders-v5` beats the baseline.

To visualize a trained agent, you can run `python3 space_invaders_demo.py <path/to/model>` or `python3 cart_pole_demo.py <path/to/model>`, depending on which game the agent was trained on. 

### Acknowledgements and Works Cited

The algorithms we implemented for this project came from Mnih et. al. and the code was adapted from an Adam Paszke and Mark Towers PyTorch tutorial. 

#### Works Cited

Mnih, Volodymyr, et al. "Playing atari with deep reinforcement learning." _arXiv preprint arXiv:1312.5602_ (2013).

Paszke, Adam & Towers, Mark (2024).  _Reinforcement learning (DQN) tutorial_. Reinforcement Learning (DQN) Tutorial - PyTorch Tutorials 2.3.0+cu121 documentation. 