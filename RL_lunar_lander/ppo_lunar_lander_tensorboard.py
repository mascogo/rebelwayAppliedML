import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os

# Directorio para logs de TensorBoard
log_dir = "./tensorboard_logs/ppo_lunarlander/"
os.makedirs(log_dir, exist_ok=True)

# Crea el entorno

# Crea el modelo con logging
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# (Opcional) Evalúa cada X pasos y guarda el mejor modelo
# eval_env = gym.make("LunarLander-v3")
eval_env =  make_vec_env("LunarLander-v3", n_envs=1)
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/best_model/",
    log_path=log_dir,
    eval_freq=5000,
    deterministic=True,
    render=False,
)

# Entrena el modelo con callback de evaluación
model.learn(total_timesteps=200000, callback=eval_callback)

# Guarda el modelo final
model.save("ppo_lunarlander")

env.close()