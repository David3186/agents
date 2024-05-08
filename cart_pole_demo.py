from agent import *
import gymnasium as gym

config = TrainConfig(
    network_dir='cartpole_gpu',
    batch_size=128,
    gamma = 0.99,
    eps_start = 0.9,
    eps_end = 0.1,
    eps_decay = 1000,
    tau = 0.005,
    lr = 1e-4,
    num_episodes = 2500,
    len_memory=10000,
)

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    network = torch.load('models/cartpole/model.pt')
    agent = Agent({'id': "CartPole-v1"}, network, num_frames=1)

    agent.train(config)

    agent.play()