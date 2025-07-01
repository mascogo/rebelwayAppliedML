import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))
env = gym.make('Acrobot-v1')  # , render_mode='human')

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

replay_buffer = deque(maxlen=100000)
print("START: replay_buffer: {}".format(len(replay_buffer)))
# Hyperparameters
batch_size = 64
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01
learning_rate = 0.001

input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n

model_path = "acrobot_model.pth"

dqn = DQN(input_dim, output_dim).to(device)
model_loaded = False
target_net = DQN(input_dim, output_dim).to(device)
if os.path.isfile(model_path):
    print("Model found in disk")
    dqn.load_state_dict(torch.load(model_path, weights_only=True))
    model_loaded = True
    print("model loaded")
    target_net.load_state_dict(torch.load(model_path, weights_only=True))
else:
    target_net.load_state_dict(dqn.state_dict())
optimizer = optim.Adam(dqn.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

def select_action(state_tensor, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()
    else:
        state_tensor = torch.FloatTensor(state_tensor).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = dqn(state_tensor)
        return q_values.argmax().item()  


# training loop
if not model_loaded:
    print("Model not found.. So.. Training!!")
    num_episodes = 1000

    for episode in range(num_episodes):
        state, _ = env.reset()
        # print("state:\n{}\n\nlen(state): {}".format(state, len(state)))
        total_reward = 0
        # print("Episode: {}")
        for t in range(1, 501):
            action = select_action(state, epsilon)
            # print("Action taken:", action)
            env_step = env.step(action)
            # print("env_step: {}".format(env_step))
            next_state, reward, done, _, _ = env_step
            replay_buffer.append((state, action, reward, next_state, done))
            # print("\tt: {}      replay_buffer: {}".format(t, len(replay_buffer)))
            state = next_state
            total_reward += reward

            if done:
                print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")
                break

        if len(replay_buffer) >= batch_size:
            batch = random.sample(replay_buffer, batch_size)
            # print("batch: {}".format(batch))
            states, actions, rewards, next_states, dones = zip(*batch)
            # print("len(states): {}".format(len(states)))
            # states_tensor = torch.FloatTensor(states).to(device)
            states_tensor = torch.tensor(np.array(states), dtype=torch.float32).to(device)
            # actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
            actions_tensor = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1).to(device)
            # rewards_tensor = torch.FloatTensor(rewards).to(device)
            rewards_tensor = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
            # next_states_tensor = torch.FloatTensor(next_states).to(device)
            next_states_tensor = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
            # dones_tensor = torch.FloatTensor(dones).to(device)
            dones_tensor = torch.tensor(np.array(dones), dtype=torch.float32).to(device)

            current_q = dqn(states_tensor).gather(1, actions_tensor).squeeze()
            next_q = dqn(next_states_tensor).max(1)[0].detach()
            target_q = rewards_tensor + (gamma * next_q * (1 - dones_tensor))

            loss = loss_fn(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            epsilon = max(epsilon, epsilon_min)

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    torch.save(dqn.state_dict(), model_path)

env = gym.make('Acrobot-v1', render_mode='human')
state, _ = env.reset()

successes = 0
for i in range(2500):
    # env.render_mode = "human"
    if i % 50 == 0:
        print("i: {}".format(i))
    env.render()
    action = select_action(state, epsilon=0)  # Use a low epsilon for testing
    state, reward, done, _, _ = env.step(action)
    if done:
        successes += 1
        print("Succeded {} times".format(successes))

    if successes > 5:
        break

env.close()
