import sys
from agent import *

def play_agent(x):
    x.play(restarts=5)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 space_invaders_demo.py <model_path>")
        sys.exit(1)

    agent = Agent({'id': "ALE/SpaceInvaders-v5", 'obs_type': 'grayscale', 'frameskip': 3}, torch.load(sys.argv[1], map_location=torch.device('cpu')))

    play_agent(agent)