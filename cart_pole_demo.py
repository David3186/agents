from agent import *
import gymnasium as gym

config = TrainConfig(
    network_dir='cartpole',
    batch_size=128,
    gamma = 0.99,
    eps_start = 0.9,
    eps_end = 0.1,
    eps_decay = 1000,
    tau = 0.005,
    lr = 1e-4,
    num_episodes = 2500
)

if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    agent = Agent({'id': "CartPole-v1"}, DQN(env.observation_space.shape[0], env.action_space.n))

    agent.train(config)

    agent.play()