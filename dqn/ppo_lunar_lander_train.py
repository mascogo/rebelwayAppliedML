import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Crea el entorno vectorizado
env = make_vec_env("LunarLander-v3", n_envs=1)

# Crea el modelo con PPO
model = PPO("MlpPolicy", env, verbose=1)

# Entrena el modelo
model.learn(total_timesteps=200_000)

# Guarda el modelo
model.save("ppo_lunarlander")