import gymnasium as gym

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")
print("env: {}".format(env))
# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
print("observation: {}",format(observation))
print("info: {}",format(info))
for i, _ in enumerate(range(1000)):
    print("i: {}".format(i))
    # this is where you would insert your policy
    action = env.action_space.sample()
    print("action: {}".format(action))
    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)
    print("observation: {}".format(observation))
    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()
    print("---------")
env.close()



