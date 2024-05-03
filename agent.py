import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from copy import deepcopy
from tqdm import tqdm

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE="cuda:2"
DEVICE=torch.device("cpu")

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
NUM_EPISODES = 500

class Agent():
    def __init__(self, game_type : str, policy_net : nn.Module):
        self.game_type = game_type
        self.policy_net = policy_net.to(DEVICE)

    def train(self):
        target_net = deepcopy(self.policy_net)

        env = gym.make(self.game_type)

        optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)

        memory = ReplayMemory(10_000)
        
        steps_done = 0

        def select_action(state):
            nonlocal steps_done
            sample = random.random()
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                math.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    # Take the argmax to compute the action to take

                    return self.policy_net(state).argmax(dim=1).unsqueeze(0)
            else:
                return torch.tensor([[env.action_space.sample()]], device=DEVICE, dtype=torch.long)
            

        def optimize_model():
            if len(memory) < BATCH_SIZE:
                return
            transitions = memory.sample(BATCH_SIZE)
            # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
            # detailed explanation). This converts batch-array of Transitions
            # to Transition of batch-arrays.
            batch = Transition(*zip(*transitions))

            # Compute a mask of non-final states and concatenate the batch elements
            # (a final state would've been the one after which simulation ended)


            non_final_mask = torch.tensor(tuple([s != None for s in batch.next_state]), device=DEVICE, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
            # columns of actions taken. These are the actions which would've been taken
            # for each batch state according to policy_net
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

            # Compute V(s_{t+1}) for all next states.
            # Expected values of actions for non_final_next_states are computed based
            # on the "older" target_net; selecting their best reward with max(1).values
            # This is merged based on the mask, such that we'll have either the expected
            # state value or 0 in case the state was final.
            next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
            with torch.no_grad():
                next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch

            # Compute Huber loss
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            # In-place gradient clipping
            torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
            optimizer.step()

        for i_episode in tqdm(range(NUM_EPISODES), leave=False, desc="Training", disable=None):
            # Initialize the environment and get its state
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            for t in count():
                action = select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=DEVICE)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)

                target_net.load_state_dict(target_net_state_dict)

                if done:
                    break

        torch.save(self.policy_net, "network2.pt")
    def get_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(state).argmax(dim=1).item()
        
    def play(self):
        env = gym.make(self.game_type, render_mode="human")
        observation, info = env.reset()
        
        while True:
            action = self.get_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
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
        self.conv1 = nn.Conv2d(3, 6, 5, stride=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, 5)
        self.fc1 = nn.Linear(6 * 17 * 23, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = x.swapdims(1, -1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        return self.fc1(x)
    

if __name__ == "__main__":
    env = gym.make("ALE/SpaceInvaders-v5")
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy_net = CNN(n_actions)
    agent = Agent("ALE/SpaceInvaders-v5", torch.load('network2.pt', map_location=torch.device('cpu')))
    # agent.train()
    agent.play()