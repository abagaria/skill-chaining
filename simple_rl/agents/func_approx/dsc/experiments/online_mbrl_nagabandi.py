import os
import ipdb
import time
import torch
import pickle
import random
import argparse
import numpy as np
from copy import deepcopy
from collections import deque
from simple_rl.agents.func_approx.dsc.experiments.utils import *
from simple_rl.tasks.d4rl_ant_maze.D4RLAntMazeMDPClass import D4RLAntMazeMDP

from simple_rl.agents.func_approx.dsc.dynamics.mpc import MPC
from simple_rl.agents.func_approx.td3.TD3AgentClass import TD3


class OnlineModelBasedSkillChaining(object):
    def __init__(self, warmup_episodes, max_steps, use_vf, stronger_baseline,
                 use_diverse_starts, use_dense_rewards, use_optimal_sampler, lr_c, lr_a,
                 experiment_name, device, logging_freq, evaluation_freq, seed):
        
        self.device = device
        self.use_vf = use_vf
        self.experiment_name = experiment_name
        self.warmup_episodes = warmup_episodes
        self.max_steps = max_steps
        self.use_diverse_starts = use_diverse_starts
        self.use_dense_rewards = use_dense_rewards
        self.use_optimal_sampler = use_optimal_sampler
        self.stronger_baseline = stronger_baseline

        self.seed = seed
        self.logging_freq = logging_freq
        self.evaluation_freq = evaluation_freq

        self.mdp = D4RLAntMazeMDP("umaze", goal_state=np.array((0, 8)), seed=seed)
        self.mpc = MPC(mdp=self.mdp,
                        state_size=self.mdp.state_space_size(),
                        action_size=self.mdp.action_space_size(),
                        dense_reward=self.use_dense_rewards,
                        device=self.device)
        if self.use_vf:
            self.value_learner = TD3(state_dim=self.mdp.state_space_size()+2,
                                    action_dim=self.mdp.action_space_size(),
                                    max_action=1.,
                                    name="td3-agent",
                                    device=self.device,
                                    lr_c=lr_c, lr_a=lr_a,
                                    use_output_normalization=True)
        
        self.target_salient_event = self.mdp.get_original_target_events()[0]
        self.goal = self.target_salient_event.get_target_position()

        self.log = {}

    def act(self, state):
        if random.random() < 0.1:
            return self.mdp.sample_random_action()
        
        vf = self.value_function if self.use_vf else None
        return self.mpc.act(state, self.goal, vf=vf)

    def value_function(self, states, goals):
        assert isinstance(states, np.ndarray)
        assert isinstance(goals, np.ndarray)

        if len(states.shape) == 1:
            states = states[None, ...]
        if len(goals.shape) == 1:
            goals = goals[None, ...]

        goal_positions = goals[:, :2]
        augmented_states = np.concatenate((states, goal_positions), axis=1)
        augmented_states = torch.as_tensor(augmented_states).float().to(self.device)
        
        values = self.value_learner.get_values(augmented_states)

        return values

    def get_augmented_state(self, state, goal):
        return np.concatenate((state.features(), goal))

    def experience_replay(self, trajectory, goal):
        for state, action, reward, next_state in trajectory:
            augmented_state = self.get_augmented_state(state, goal)
            augmented_next_state = self.get_augmented_state(next_state, goal)

            reward_func = self.mdp.dense_gc_reward_function if self.use_dense_rewards \
                else self.mdp.sparse_gc_reward_function
            reward, global_done = reward_func(next_state, goal, info={})

            self.value_learner.step(augmented_state, action, reward, augmented_next_state, global_done)

    def random_rollout(self, num_steps):
        step_number = 0
        state = deepcopy(self.mdp.cur_state)
        while step_number < num_steps:
            state = deepcopy(self.mdp.cur_state)
            action = self.mdp.sample_random_action()
            reward, next_state = self.mdp.execute_agent_action(action)
            self.mpc.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
            step_number += 1
        return step_number

    def rollout(self, num_steps):
        step_number = 0
        transitions = []
        state = deepcopy(self.mdp.cur_state)
        while step_number < num_steps and not self.mdp.sparse_gc_reward_function(state, self.goal, {})[1]:
            state = deepcopy(self.mdp.cur_state)
            if step_number % 200 == 0:
                print(f"[Step: {step_number}] Rolling out from {state.position} targeting {self.goal}")
            action = self.act(state)
            reward, next_state = self.mdp.execute_agent_action(action)
            transitions.append((state, action, reward, next_state))
            self.mpc.step(state.features(), action, reward, next_state.features(), next_state.is_terminal())
            step_number += 1

        if self.use_vf:
            self.experience_replay(transitions, self.goal)

        return step_number

    def run_loop(self, num_episodes, num_steps, start_episode=0):
        per_episode_durations = []
        last_10_durations = deque(maxlen=10)

        for episode in range(start_episode, start_episode + num_episodes):
            self.reset(episode)

            if self.stronger_baseline:
                self.goal = self.mdp.get_position(self.mdp.sample_random_state())

            step = self.rollout(num_steps) if episode > self.warmup_episodes else self.random_rollout(num_steps)

            last_10_durations.append(step)
            per_episode_durations.append(step)
            self.log_status(episode, last_10_durations)

            if episode == self.warmup_episodes - 1:
                self.learn_dynamics_model(epochs=50)
            elif episode >= self.warmup_episodes:
                self.learn_dynamics_model(epochs=5)

            self.log_success_metrics(episode)

        return per_episode_durations

    def log_success_metrics(self, episode):
        if episode % self.evaluation_freq == 0 and episode > self.warmup_episodes:
            success, step_count = test_agent(self, 1, self.max_steps)

            self.log[episode] = {}
            self.log[episode]["success"] = success
            self.log[episode]["step-count"] = step_count[0]

            with open(f"{self.experiment_name}/log_file_{self.seed}.pkl", "wb+") as log_file:
                pickle.dump(self.log, log_file)

    def learn_dynamics_model(self, epochs=50, batch_size=1024):
        self.mpc.load_data()
        self.mpc.train(epochs=epochs, batch_size=batch_size)

    def log_status(self, episode, last_10_durations):
        print(f"Episode {episode} \t Mean Duration: {np.mean(last_10_durations)}")

        if episode % self.logging_freq == 0 and episode != 0:
            self.goal = self.target_salient_event.get_target_position()
            if self.use_vf:
                make_chunked_goal_conditioned_value_function_plot(self.value_learner,
                                                                goal=self.goal,
                                                                episode=episode, seed=self.seed,
                                                                experiment_name=self.experiment_name)

    def reset(self, episode):
        self.mdp.reset()

        if self.use_diverse_starts and episode > self.warmup_episodes:
            random_state = self.mdp.sample_random_state()
            random_position = self.mdp.get_position(random_state)
            self.mdp.set_xy(random_position)


def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path

def test_agent(exp, num_experiments, num_steps):
    def rollout():
        step_number = 0
        while step_number < num_steps and not exp.mdp.sparse_gc_reward_function(exp.mdp.cur_state, exp.goal, {})[1]:
            state = deepcopy(exp.mdp.cur_state)
            action = exp.act(state)
            exp.mdp.execute_agent_action(action)
            step_number += 1
        return step_number
        
    success = 0
    step_counts = []
    exp.goal = exp.target_salient_event.get_target_position()

    for _ in tqdm(range(num_experiments), desc="Performing test rollout"):
        exp.mdp.reset()
        steps_taken = rollout()
        if steps_taken != num_steps:
            success += 1
        step_counts.append(steps_taken)

    print("*" * 80)
    print(f"Test Rollout Success Rate: {success / num_experiments}, Duration: {np.mean(step_counts)}")
    print("*" * 80)

    return success / num_experiments, step_counts

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--episodes", type=int, default=150)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--warmup_episodes", type=int, default=5)
    parser.add_argument("--use_value_function", action="store_true", default=False)
    parser.add_argument("--use_diverse_starts", action="store_true", default=False)
    parser.add_argument("--use_dense_rewards", action="store_true", default=False)
    parser.add_argument("--use_optimal_sampler", action="store_true", default=False)
    parser.add_argument("--logging_frequency", type=int, default=50, help="Draw init sets, etc after every _ episodes")
    parser.add_argument("--evaluation_frequency", type=int, default=10)
    parser.add_argument("--stronger_baseline", action="store_true", default=False)

    parser.add_argument("--lr_c", type=float, help="critic learning rate")
    parser.add_argument("--lr_a", type=float, help="actor learning rate")
    args = parser.parse_args()

    assert args.use_value_function or args.use_dense_rewards

    exp = OnlineModelBasedSkillChaining(experiment_name=args.experiment_name,
                                        device=torch.device(args.device),
                                        warmup_episodes=args.warmup_episodes,
                                        max_steps=args.steps,
                                        use_vf=args.use_value_function,
                                        use_diverse_starts=args.use_diverse_starts,
                                        use_dense_rewards=args.use_dense_rewards,
                                        use_optimal_sampler=args.use_optimal_sampler,
                                        logging_freq=args.logging_frequency,
                                        evaluation_freq=args.evaluation_frequency,
                                        stronger_baseline=args.stronger_baseline,
                                        seed=args.seed,
                                        lr_c=args.lr_c,
                                        lr_a=args.lr_a)

    create_log_dir(args.experiment_name)
    create_log_dir(f"initiation_set_plots/{args.experiment_name}")
    create_log_dir(f"value_function_plots/{args.experiment_name}")

    start_time = time.time()
    durations = exp.run_loop(args.episodes, args.steps)
    end_time = time.time()

    # print("exporting graphs!")
    # exp.log_status(2000, [1000])

    print("TIME: ", end_time - start_time)

