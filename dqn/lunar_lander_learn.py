import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from torch.utils.bundled_inputs import bundle_randn

# Create environment
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Instantiate the agent
model = DQN('MlpPolicy', env, learning_rate=1e-3, verbose=1)
# model = PPO("MlpPolicy", env, verbose=1)
# Train the agent and display a progress bar

def evaluate(model, num_steps=1000):
    episode_rewards = [0.0]
    obs, _info = env.reset()
    for i in range(num_steps):
        action, _states = model.predict(obs)

        obs, reward, done, info, _ = env.step(action)

        episode_rewards[-1] += reward
        if done:
            obs, _info = env.reset()
            episode_rewards.append(0.0)

    mean_100ep_reward = round(np.mean(episode_rewards[-100:]), 1)
    print("Mean record: {}\tNum episodes: {}".format(mean_100ep_reward, len(episode_rewards)))

mean_reward_before_training = evaluate(model, num_steps=10000)

model.learn(total_timesteps=200000, progress_bar=True, log_interval = 10)

model.save("dqn_lunar_dqn2")
del model


model = DQN.load("dqn_lunar_dqn2")

mean_reward = evaluate(model, num_steps=10000)

# episodes = 10
#
# for ep in range(episodes):
#     obs = env.reset()
#     done = False
#     while not done:
#         env.render()
#         t = env.step(env.action_space.sample())
#         print("t: {}".format(t))
#         obs, reward, done, info, _ = t
#
# env.close()
# Save the agent

# model.save("dqn_lunar_ppo")