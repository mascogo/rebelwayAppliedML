import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

# Carpetas
model_path = "models/best_model/best_model"
video_folder = "./videos/final_run/"
os.makedirs(video_folder, exist_ok=True)

# Carga el modelo
model = PPO.load(model_path)

# Crea entorno y vectoriza
env = DummyVecEnv([lambda: gym.make("LunarLander-v3",  render_mode="rgb_array")])

# Graba un solo vídeo de 1000 steps
env = VecVideoRecorder(
    env,
    video_folder,
    record_video_trigger=lambda step: step == 0,  # solo al inicio
    video_length=20000,
    name_prefix="ppo_lunarlander_final"
)

# Simulación
obs = env.reset()
for _ in range(10000):
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()

print(f"✅ Vídeo guardado en: {os.path.abspath(video_folder)}")