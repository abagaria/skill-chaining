import ipdb
import random
import torch
import argparse
import torch.nn
import numpy
import numpy as np
from copy import deepcopy
from collections import deque

import pfrl
from pfrl import explorers, replay_buffers
from pfrl import nn as pnn
from pfrl.q_functions import DistributionalDuelingDQN

from simple_rl.tasks.gym.GymMDPClass import GymMDP
from simple_rl.tasks.gym.GymStateClass import GymState as State
from simple_rl.tasks.gym.wrappers import LazyFrames
from simple_rl.agents.func_approx.rainbow.hyperparameters import *


class RainbowAgent(object):
    def __init__(self, obs_size, n_actions,
                random_action_func,
                pixel_observation=True,
                gamma=0.99,
                noisy_net_sigma=0.5,
                buffer_size=int(3e5),
                hyperparameter_setting="efficient",
                goal_conditioning=False,
                device_id=0):

        assert hyperparameter_setting in ("rainbow", "efficient"), hyperparameter_setting
        params = rainbow_params if hyperparameter_setting == "rainbow" else efficient_rainbow_params
        print(f"Using the following hyperparameters for Rainbow agent: {params}")
        print(f"Using device with ID {device_id} for Rainbow agent")

        # Q-Function architecture
        n_input_channels = 4 + int(goal_conditioning)
        q_func = DistributionalDuelingDQN(n_actions,
                                          n_input_channels=n_input_channels,
                                          n_atoms=51,
                                          v_min=-10.,
                                          v_max=+10.)
        
        # Adam optimizer
        optimizer = torch.optim.Adam(q_func.parameters(), params['lr'], eps=1.5 * 10 ** -4)

        # Noisy nets
        pnn.to_factorized_noisy(q_func, sigma_scale=noisy_net_sigma)

        # Turn off explorer
        explorer = explorers.Greedy()

        # Prioritized experience replay
        betasteps = params['betasteps']
        replay_buffer = replay_buffers.PrioritizedReplayBuffer(capacity=buffer_size,
                                                               alpha=0.5,
                                                               beta0=0.4,
                                                               betasteps=betasteps,
                                                               num_steps=params['num_steps'],
                                                               normalize_by_max="memory")
        # Feature extractor
        def phi(x):
            return np.asarray(x, dtype=np.float32) / 255.

        # Set the device id to use GPU. To use CPU only, set it to -1.
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}" if device_id > -1 else "cpu")

        # Now create an agent that will interact with the environment.
        self.agent = pfrl.agents.CategoricalDoubleDQN(q_function=q_func,
                                                      optimizer=optimizer, 
                                                      replay_buffer=replay_buffer,
                                                      gamma=gamma,
                                                      explorer=explorer,
                                                      minibatch_size=32,
                                                      replay_start_size=params['replay_start_size'],
                                                      target_update_interval=params['target_update_interval'],
                                                      update_interval=params['update_interval'],
                                                      batch_accumulator="mean",
                                                      phi=phi, 
                                                      gpu=self.device_id)
    
    def act(self, obs):
        return self.agent.act(obs)
    
    def step(self, obs, action, reward, next_obs, done):
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


def random_step(mdp):
        state = deepcopy(mdp.cur_state)
        action = mdp.sample_random_action()
        reward, next_state = mdp.execute_agent_action(action)
        return state, action, reward, next_state

def learning_step(mdp, agent, goal):
    assert goal is not None 

    state = deepcopy(mdp.cur_state)  # TODO: Does this undo the impact of LazyFrames?
    obs = state.features()
    augmented_state = agent.get_augmented_state(obs, goal)

    action = agent.act(augmented_state)
    reward, next_state = mdp.execute_agent_action(action)
    return state, action, reward, next_state

def train(agent, mdp, max_steps):
    
    def experience_replay(traj):
        """ Vanilla episodic experience replay. """
        for s, a, r, sp, d in traj:
            agent.step(s, a, r, sp, d)
        
    reset(mdp, diverse_starts=True)

    episode = 0
    trajectory = []
    episode_length = 0
    episode_reward = 0
    per_episode_rewards = []
    
    for step_number in range(max_steps):
        obs = mdp.cur_state.features()
        action = agent.act(obs)
        reward, next_state = mdp.execute_agent_action(action)
        next_obs = next_state.features()
        
        # trajectory.append((obs, action, reward, next_obs, next_state.is_terminal()))
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

            # experience_replay(trajectory)
            
            trajectory = []
            reset(mdp, diverse_starts=True)

    return per_episode_rewards


def her_train(agent, mdp, max_steps):

    def experience_replay(traj, g):
        """ Goal conditioned episodic experience replay. """
        for s, a, _, sp in traj:
            r, d = mdp.sparse_gc_reward_function(sp, g)
            sg = agent.get_augmented_state(s, g)
            spg = agent.get_augmented_state(sp, g)
            agent.step(sg, a, r, spg, d)

    reset(mdp, diverse_starts=True)
    goal_buffer = deque(maxlen=10)

    goal = None
    episode = 0
    trajectory = []
    episode_length = 0
    episode_reward = 0
    per_episode_rewards = []

    for step_number in range(max_steps):
        transition = learning_step(mdp, agent, goal) if len(goal_buffer) > 0 else random_step(mdp)
        
        # Logging
        reward, next_state = transition[2], transition[3]
        episode_length += 1
        episode_reward += reward
        trajectory.append(transition)
        
        if next_state.is_terminal():
            per_episode_rewards.append(episode_reward)
            print(f"Episode: {episode} \t Step Number: {step_number} \t Episode length = {episode_length} \t Reward = {episode_reward}")

            if goal is not None:
                print(f"Performing experience replay on pursued goal {goal.position}")
                experience_replay(trajectory, goal)
                
                print(f"Performing experience replay on reached goal {next_state.position}")
                experience_replay(trajectory, next_state)

                print(f"Done performing experience replay on {len(trajectory)} transitions.")

            episode += 1
            trajectory = []
            episode_length = 0
            episode_reward = 0

            if mdp.is_goal_state(next_state):
                goal_buffer.append(next_state)

            if len(goal_buffer) > 0:
                goal = random.choice(goal_buffer)
                print(f"Setting {goal.position} as the new goal")
            
            reset(mdp, diverse_starts=True) 

    return per_episode_rewards


def test(agent, mdp, max_steps, diverse_starts):
    reset(mdp, diverse_starts=diverse_starts)

    episode = 0
    episode_length = 0
    episode_reward = 0
    per_episode_rewards = []
    
    for step_number in range(max_steps):
        obs = mdp.cur_state.features()
        action = agent.act(obs)
        reward, next_state = mdp.execute_agent_action(action)

        # Logging
        episode_length += 1
        episode_reward += reward
        
        if next_state.is_terminal():
            print(f"Episode: {episode} \t Step Number: {step_number} \t Episode length = {episode_length} \t Reward = {episode_reward}")
            episode += 1
            per_episode_rewards.append(episode_reward)
            episode_length = 0
            episode_reward = 0
            
            reset(mdp, diverse_starts=diverse_starts)

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
    args = parser.parse_args()

    mdp = GymMDP(env_name="MontezumaRevenge", pixel_observation=True, render=args.render, seed=1)
    obs_size = mdp.state_space_size()
    n_actions = mdp.action_space_size()

    agent = RainbowAgent(obs_size, n_actions,
                         goal_conditioning=args.use_her,
                         pixel_observation=True,
                         random_action_func=mdp.sample_random_action, 
                         hyperparameter_setting=args.param_setting)

    if args.use_her:
        training_rewards = her_train(agent, mdp, max_steps=args.steps)
    else:
        training_rewards = train(agent, mdp, max_steps=args.steps)
