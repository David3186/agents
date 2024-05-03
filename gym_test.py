import gymnasium as gym
# env = gym.make("ALE/SpaceInvaders-v5", render_mode="human")
env = gym.make("ALE/SpaceInvaders-v5")
observation, info = env.reset(seed=42)
from time import time 
t0 = time()
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

print("Time taken: ", time() - t0)

env.close()