import torch
import numpy as np
from simple_rl.agents.func_approx.opiq.env import make_montezuma
from simple_rl.agents.func_approx.opiq.dqn import OptimisticDQN

NUM_TRAINING_STEPS = int(1e7)
MAX_EPISODE_LENGTH = int(1e4)

def make_agent(env, device):
    return OptimisticDQN(input_shape=env.observation_space.shape,
                         num_actions=env.action_space.n,
                         device=device,
                         count_representation_shape=(8, 11),
                         eps_start=1., eps_finish=0.01, eps_length=1000000,
                         optim_m=2,
                         optim_beta=0.1,
                         optim_action_tau=0.1,
                         optim_bootstrap_tau=0.01,
                         lr=5e-4,  # TODO: Verify
                         max_grad_norm=5.,  # TODO: Verify
                         update_interval=4,
                         batch_size=32,
                         max_buffer_len=int(1e6),
                         n_steps=3,
                         gamma=0.99)

env = make_montezuma()
agent = make_agent(env, torch.device("cpu"))
obs = env.reset()

episode_reward = 0.
cumulative_reward = 0.
episodic_rewards = []
episode_length = 0

for step_number in range(NUM_TRAINING_STEPS):
    action = agent.act(obs)
    agent.visit(obs, action)
    next_obs, reward, done, info = env.step(action)
    agent.update(obs, action, reward, next_obs, done)

    obs = next_obs
    episode_length += 1
    episode_reward += reward
    cumulative_reward += reward

    if done or episode_length >= MAX_EPISODE_LENGTH:
        print(f"Step: {step_number} | Score: {episode_reward} | Cum Score: {cumulative_reward}")
        episodic_rewards.append(episode_reward)
        episode_reward = 0
        episode_length = 0
        obs = env.reset()

    if step_number % 8000 == 0:
        agent.sync_target_network()
