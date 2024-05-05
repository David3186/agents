from matplotlib import pyplot as plt
from pathlib import Path
import time
import numpy as np
import sys
import gymnasium as gym
from tqdm import tqdm

def get_baseline_reward(game_type):
    env = gym.make(game_type)
    # Approximate reward of random policy by running the environment
    rewards = []
    for _ in tqdm(range(100)):
        observation, info = env.reset()
        total_reward = 0
        for _ in range(1000):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        rewards.append(total_reward)
    return np.mean(rewards)

def plot_running_avg(data, output_dir: Path, baseline_reward=None):
    output_dir = Path(output_dir)
    # Plot data as blue and running average of past 50 episodes as red
    N = len(data)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(data[max(0, t-50):(t+1)])
    plt.plot(data, label='Episode Lengths', alpha=0.3)
    if baseline_reward is not None:
        plt.plot([baseline_reward]*N, label='Baseline', linestyle='--')
    plt.plot(running_avg, label='Running Average')
    plt.title("Running Average of Episode Lengths")
    plt.savefig(output_dir / "running_avg.png")
    plt.clf()


if __name__ == "__main__":
    outfile = Path(sys.argv[1])
    baseline_reward = get_baseline_reward("ALE/SpaceInvaders-v5")
    while True:
        data = [float(x) for x in Path(outfile).read_text().strip().split("\n")]
        output_dir = Path(sys.argv[2])
        plot_running_avg(data, output_dir, baseline_reward)
        time.sleep(15)