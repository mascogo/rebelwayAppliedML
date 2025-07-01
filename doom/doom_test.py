import gymnasium
from vizdoom import gymnasium_wrapper
env = gymnasium.make("VizdoomDeadlyCorridor-v0")
observation, info = env.reset()
for _ in range(1000):
   action = policy(observation)  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()