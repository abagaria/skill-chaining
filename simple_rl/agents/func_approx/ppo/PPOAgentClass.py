import ipdb
import random
import argparse
import numpy as np
from copy import deepcopy
from collections import deque

import torch
from torch import nn

import pfrl
from pfrl.agents import PPO
from pfrl.policies import SoftmaxCategoricalHead

from simple_rl.tasks.gym.GymMDPClass import GymMDP
from simple_rl.tasks.gym.GymStateClass import GymState as State
from simple_rl.tasks.gym.wrappers import LazyFrames


class PPOAgent(object):
    def __init__(self,
                 obs_n_channels,
                 n_actions, 
                 lr=2.5e-4,
                 update_interval=128*8,
                 batchsize=32*8,
                 epochs=4,
                 device_id=0):

        def lecun_init(layer, gain=1):
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                pfrl.initializers.init_lecun_normal(layer.weight, gain)
                nn.init.zeros_(layer.bias)
            else:
                pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
                pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
                nn.init.zeros_(layer.bias_ih_l0)
                nn.init.zeros_(layer.bias_hh_l0)
            return layer

        self.model = nn.Sequential(
            lecun_init(nn.Conv2d(obs_n_channels, 32, 8, stride=4)),
            nn.ReLU(),
            lecun_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            lecun_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            lecun_init(nn.Linear(3136, 512)),
            nn.ReLU(),
            pfrl.nn.Branched(
                nn.Sequential(
                    lecun_init(nn.Linear(512, n_actions), 1e-2),
                    SoftmaxCategoricalHead(),
                ),
                lecun_init(nn.Linear(512, 1)),
            ),
        )

        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}" if device_id > -1 else "cpu")
        opt = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)

        def phi(x):
            # Feature extractor
            return np.asarray(x, dtype=np.float32) / 255.

        self.agent = PPO(self.model,
                         opt,
                         gpu=device_id,
                         phi=phi,
                         update_interval=update_interval,
                         minibatch_size=batchsize,
                         epochs=epochs,
                         clip_eps=0.1,
                         clip_eps_vf=None,
                         standardize_advantages=True,
                         entropy_coef=1e-2,
                         recurrent=False,
                         max_grad_norm=0.5,
        )

    def act(self, obs):
        return self.agent.act(obs)
    
    def step(self, obs, action, reward, next_obs, done):
        ipdb.set_trace()
        self.agent.observe(next_obs, reward, done, reset=False)

    def get_batched_qvalues(self, states):
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        
        device = f"cuda:{self.device_id}" if self.device_id >= 0 else "cpu"
        states = states.to(device)
        with torch.no_grad():
            action_values = self.agent.model(states)
        return action_values.q_values

    @staticmethod
    def get_augmented_state(state, goal):
        assert isinstance(state, (State, LazyFrames)), f"{type(state)}"
        assert isinstance(goal, (State, LazyFrames)), f"{type(goal)}"

        state = state.features() if isinstance(state, State) else state
        goal = goal.features() if isinstance(goal, State) else goal
        
        obs_frames = state._frames  # shallow copy
        goal_frame = goal.get_frame(-1)  # last frame of the goal image stack
        augmented_frames = obs_frames + [goal_frame]

        assert len(augmented_frames) == 5, len(augmented_frames)
        assert len(obs_frames) == 4, len(obs_frames)

        return LazyFrames(frames=augmented_frames, stack_axis=0)


def train(agent, mdp, max_steps):
    
    reset(mdp, diverse_starts=True)

    episode = 0
    episode_length = 0
    episode_reward = 0
    per_episode_rewards = []
    
    for step_number in range(max_steps):
        obs = mdp.cur_state.features()
        action = agent.act(obs)
        reward, next_state = mdp.execute_agent_action(action)
        next_obs = next_state.features()

        agent.step(obs, action, reward, next_obs, next_state.is_terminal())

        # Logging
        episode_length += 1
        episode_reward += reward
        
        if next_state.is_terminal():
            print(f"Episode: {episode} \t Step Number: {step_number} \t Episode length = {episode_length} \t Reward = {episode_reward}")
            episode += 1
            per_episode_rewards.append(episode_reward)
            episode_length = 0
            episode_reward = 0

            reset(mdp, diverse_starts=True)

    return per_episode_rewards

def random_step(mdp):
    """ We cannot learn from random transitions for PPO. """
    action = mdp.sample_random_action()
    reward, next_state = mdp.execute_agent_action(action)
    return reward, next_state

def learning_step(mdp, agent, goal):
    """ Pick and action and use the resulting reward, next-state to update PPO. """
    assert goal is not None 

    obs = mdp.cur_state.features()
    augmented_state = agent.get_augmented_state(obs, goal)

    action = agent.act(augmented_state)
    reward, next_state = mdp.execute_agent_action(action)

    next_obs = next_state.features()
    augmented_next_obs = agent.get_augmented_state(next_obs, goal)
    agent.agent.observe(augmented_next_obs, reward, next_state.is_terminal(), False)

    return reward, next_state

def gc_train(agent, mdp, max_steps):
        
    reset(mdp, diverse_starts=True)
    goal_buffer = deque(maxlen=10)

    goal = None
    episode = 0
    episode_length = 0
    episode_reward = 0
    per_episode_rewards = []
    
    for step_number in range(max_steps):
        reward, next_state = learning_step(mdp, agent, goal) if len(goal_buffer) > 0 else random_step(mdp)

        # Logging
        episode_length += 1
        episode_reward += reward
        
        if next_state.is_terminal():
            print(f"Episode: {episode} \t Step Number: {step_number} \t Episode length = {episode_length} \t Reward = {episode_reward}")
            episode += 1
            per_episode_rewards.append(episode_reward)
            episode_length = 0
            episode_reward = 0

            if mdp.is_goal_state(next_state):
                goal_buffer.append(next_state)

            if len(goal_buffer) > 0:
                goal = random.choice(goal_buffer)
                print(f"Setting {goal.position} as the new goal")

            reset(mdp, diverse_starts=True)

    return per_episode_rewards

def reset(mdp, diverse_starts=True):
    mdp.reset()

    if diverse_starts:
        x, y = random.choice(mdp.spawn_states)
        mdp.set_player_position(x, y)
        for _ in range(4): mdp.execute_agent_action(0)
    
    print(f"Starting policy rollout from {mdp.cur_state.position}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
    parser.add_argument("--steps", type=int, default=int(10**7))
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--param_setting", type=str, default="efficient")
    parser.add_argument("--use_her", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mdp = GymMDP(env_name="MontezumaRevenge", pixel_observation=True, render=args.render, seed=args.seed)
    obs_size = mdp.state_space_size()
    n_actions = mdp.action_space_size()

    agent = PPOAgent(5, n_actions)
    training_rewards = gc_train(agent, mdp, max_steps=args.steps)
