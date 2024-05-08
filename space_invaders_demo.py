import sys
from agent import *

def play_agent(x):
    x.play(restarts=5)

if __name__ == "__main__":
    env = gym.make("ALE/SpaceInvaders-v5", obs_type="grayscale", frameskip=3)
    n_observations = env.observation_space.shape[0]
    n_actions = env.action_space.n
    policy_net = CNN(n_actions)

    random_agent = Agent({'id': "ALE/SpaceInvaders-v5", 'obs_type': 'grayscale', 'frameskip': 3}, policy_net)

    agent = Agent({'id': "ALE/SpaceInvaders-v5", 'obs_type': 'grayscale', 'frameskip': 3}, torch.load(sys.argv[1], map_location=torch.device('cpu')))

    import multiprocessing

    p1 = multiprocessing.Process(target=play_agent, args=(random_agent,))
    p2 = multiprocessing.Process(target=play_agent, args=(agent,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    # for _ in range(5):
        # agent.play()