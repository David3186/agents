from dataclasses import dataclass
import plot_running_avg
import json
from pathlib import Path
import sys
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from functools import partial

from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy
from tqdm import tqdm

print = partial(print, flush=True)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE="cuda:1" 
# DEVICE=torch.device("cpu")

@dataclass
class TrainConfig:
    batch_size: int
    gamma: float
    eps_start: float
    eps_end: float
    eps_decay: int
    tau: float
    lr: float
    num_episodes: int

CNN_DIR = "context_paper_network_v2"

config = TrainConfig(
    batch_size=128,
    gamma = 0.99,
    eps_start = 0.9,
    eps_end = 0.1,
    eps_decay = 1000,
    tau = 0.005,
    lr = 1e-4,
    num_episodes = 100_000
)

class Agent():
    def __init__(self, game_type : str, policy_net : nn.Module):
        self.game_type = game_type
        self.policy_net = policy_net.to(DEVICE)
        

    def train(self, train_config: TrainConfig):
        self.baseline_reward = plot_running_avg.get_baseline_reward(self.game_type)
        model_dir = ("models" / Path(CNN_DIR))
        if model_dir.exists():
            if len(sys.argv) <= 1 or not sys.argv[1] == "--force":
                raise FileExistsError(f"Model directory {model_dir} already exists")
        model_dir.mkdir(exist_ok=True)
        (model_dir / 'train_config.json').write_text(json.dumps(train_config.__dict__, indent=2))

        CNN_FILE = model_dir / "model.pt"

        target_net = deepcopy(self.policy_net)

        env = gym.make(self.game_type, obs_type="grayscale", frameskip=3)

        optimizer = optim.AdamW(self.policy_net.parameters(), lr=train_config.lr, amsgrad=True)

        memory = ReplayMemory(20_000)
        
        steps_done = 0
        data = []

        def select_action(state):
            nonlocal steps_done
            sample = random.random()
            eps_threshold = train_config.eps_end + (train_config.eps_start - train_config.eps_end) * math.exp(-steps_done / train_config.eps_decay)
            steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # Take the argmax to compute the action to take
                    state = torch.cat(tuple(state)).unsqueeze(0)
                    return self.policy_net(state).argmax(dim=1).unsqueeze(0)
            else:
                return torch.tensor([[env.action_space.sample()]], device=DEVICE, dtype=torch.long)
            

        def optimize_model():
            if len(memory) < train_config.batch_size:
                return
            transitions = memory.sample(train_config.batch_size)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))
            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)


            non_final_mask = torch.tensor(tuple([s != None for s in batch.next_state]), device=DEVICE, dtype=torch.bool)
            non_final_next_states = torch.stack([s for s in batch.next_state if s is not None])

            state_batch = torch.stack(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            # import pdb; pdb.set_trace()
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1).values
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(train_config.batch_size, device=DEVICE)
            with torch.no_grad():
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * train_config.gamma) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            optimizer.step()

        for i_episode in tqdm(range(train_config.num_episodes), leave=False, desc="Training", disable=None):
            
            # Initialize the environment and get its state
            rewards = []
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            if i_episode % 10 == 0:
                torch.save(self.policy_net, CNN_FILE)
                plot_running_avg.plot_running_avg(data, model_dir, self.baseline_reward)

            state = deque(maxlen=4)

            for t in count():
                action = select_action(state) if len(state) >= 4 else torch.tensor([[env.action_space.sample()]], device=DEVICE, dtype=torch.long)

                observation, reward, terminated, truncated, _ = env.step(action.item())
                rewards.append(reward)
                reward = torch.tensor([reward], device=DEVICE)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    # next_state = deque((state[1:], torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)), 4)
                    next_state = deepcopy(state)
                    next_state.append(torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0))

                # Store the transition in memory
                if len(state) >= 4: 
                    memory.push(torch.cat(tuple(state)), action, torch.cat(tuple(next_state)) if next_state else None, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*train_config.tau + target_net_state_dict[key]*(1-train_config.tau)

                target_net.load_state_dict(target_net_state_dict)

                if done:
                    with open(model_dir / "rewards.txt", "a") as outfile:
                        print(f"Rewards: {sum(rewards)}", file=outfile)
                        print(f"Cum steps: {steps_done}", file=outfile)
                        

                    data.append(sum(rewards))
                    break

        
        torch.save(self.policy_net, CNN_FILE)

    def get_action(self, state):
        with torch.no_grad():
            state = torch.cat(state).unsqueeze(0)
            return self.policy_net(state).argmax(dim=1).item()
        
    def play(self):
        env = gym.make(self.game_type, obs_type="grayscale", render_mode="human", frameskip=3)
        observation, info = env.reset()
        
        state = deque(maxlen=4)
        state.append(observation)

        while True:
            action = self.get_action(state) if len(state) >= 4 else env.action_space.sample()
            
            observation, reward, terminated, truncated, info = env.step(action)
            
            observation = torch.tensor(observation, dtype=torch.float32, device=DEVICE)
            state.append(observation)
            
            if terminated or truncated:
                break

class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class CNN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(13824, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    

if __name__ == "__main__":
    env = gym.make("ALE/SpaceInvaders-v5", obs_type="grayscale", frameskip=3)
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy_net = CNN(n_actions)
    agent = Agent("ALE/SpaceInvaders-v5", policy_net)
    agent.train(config)
    # agent = Agent("ALE/SpaceInvaders-v5", torch.load("models/supercracked_macronetwork/model.pt"))
    agent.play()