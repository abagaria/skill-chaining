# Python imports.
from collections import deque, defaultdict
from copy import deepcopy
import argparse
import os
import pdb
import numpy as np
from sklearn import svm

# Other imports.
from simple_rl.agents.func_approx.dsc.SkillChainingAgentClass import SkillChaining
from simple_rl.tasks.point_maze.PortablePointMazeMDPClass import PortablePointMazeMDP


class PortableSkillChainingAgent(object):
    def __init__(self, train_mdp_1, train_mdp_2, test_mdp, train_episodes, train_steps, test_episodes, test_steps, lr_actor, lr_critic,
                 ddpg_batch_size, device, max_num_options=5, subgoal_reward=0., enable_option_timeout=True,
                 buffer_length=20, num_subgoal_hits_required=3, experiment_name="portable_skill_experiment",
                 classifier_type="ocsvm", generate_plots=False, use_full_smdp_update=False,
                 log_dir="", seed=0, tensor_log=False, num_training_episodes=750, num_training_steps=2000,
                 num_test_episodes=750, num_test_steps=2000):

        # Enumerate input params for our Skill Chaining Agents
        self.train_dsc_params_1 = (train_mdp_1, train_episodes, train_steps, lr_actor, lr_critic, ddpg_batch_size, device,
                                   experiment_name, max_num_options, subgoal_reward, enable_option_timeout, buffer_length,
                                   num_subgoal_hits_required, classifier_type, generate_plots, use_full_smdp_update,
                                   log_dir, seed, tensor_log)
        self.test_dsc_params = (test_mdp, test_episodes, test_steps, lr_actor, lr_critic, ddpg_batch_size, device,
                           experiment_name, max_num_options, subgoal_reward, enable_option_timeout, buffer_length,
                           num_subgoal_hits_required, classifier_type, generate_plots, use_full_smdp_update,
                           log_dir, seed, tensor_log)

        # Create our train + test agents
        self.dsc_agent_1 = SkillChaining(*self.train_dsc_params_1)
        self.transfer_dsc_agent = SkillChaining(*self.test_dsc_params)

        # Book keeping params
        self.train_mdp_1 = train_mdp_1
        self.train_mdp_2 = train_mdp_2
        self.test_mdp = test_mdp
        self.num_training_episodes = num_training_episodes
        self.num_training_steps = num_training_steps
        self.num_test_episodes = num_test_episodes
        self.num_test_steps = num_test_steps

    def train(self):
        scores, durations = self.dsc_agent_1.skill_chaining(self.num_training_episodes, self.num_training_steps)
        return scores, durations

    def create_transfer_agent(self):
        for option in self.dsc_agent_1.trained_options:
            option.clear_for_new_task()

    def evaluate(self):
        # For logging purposes
        per_episode_scores = []
        per_episode_durations = []
        last_10_scores = deque(maxlen=10)
        last_10_durations = deque(maxlen=10)

        untrained_option_idx = 1
        self.transfer_dsc_agent.untrained_option = self.dsc_agent_1.trained_options[untrained_option_idx]

        for episode in range(self.num_test_episodes):

            self.transfer_dsc_agent.mdp.reset()
            score = 0.
            step_number = 0
            uo_episode_terminated = False
            state = deepcopy(self.transfer_dsc_agent.mdp.init_state)
            self.transfer_dsc_agent.init_states.append(deepcopy(state))
            experience_buffer = []
            state_buffer = []
            episode_option_executions = defaultdict(lambda: 0)

            while step_number < self.num_test_steps:
                experiences, reward, state, steps = self.transfer_dsc_agent.take_action(state, step_number,
                                                                                        episode_option_executions)
                score += reward
                step_number += steps
                for experience in experiences:
                    experience_buffer.append(experience)
                    state_buffer.append(experience[0])

                # Don't forget to add the last s' to the buffer
                if state.is_terminal() or (step_number == self.num_test_steps - 1):
                    state_buffer.append(state)

                if self.transfer_dsc_agent.untrained_option is not None:
                    if self.transfer_dsc_agent.untrained_option.is_term_true(state) and (not uo_episode_terminated) and \
                            self.transfer_dsc_agent.max_num_options > 0 and \
                            self.transfer_dsc_agent.untrained_option.get_training_phase() == "gestation":
                        uo_episode_terminated = True
                        if self.transfer_dsc_agent.untrained_option.train(experience_buffer, state_buffer):
                            self.transfer_dsc_agent.augment_agent_with_new_option(self.transfer_dsc_agent.untrained_option)
                    if self.transfer_dsc_agent.untrained_option.get_training_phase() == "initiation_done" \
                            or self.transfer_dsc_agent.untrained_option.completed_initiation_phase:
                        if self.transfer_dsc_agent.should_create_more_options() and \
                                untrained_option_idx < len(self.dsc_agent_1.trained_options):
                            untrained_option_idx += 1
                            new_option = self.dsc_agent_1.trained_options[untrained_option_idx]
                            print("\rTRANSFER AGENT: Setting UntrainedOption to {}".format(new_option))
                            self.transfer_dsc_agent.untrained_option = new_option

                if state.is_terminal():
                    break

            last_10_scores.append(score)
            last_10_durations.append(step_number)
            per_episode_scores.append(score)
            per_episode_durations.append(step_number)

            self.transfer_dsc_agent.log_dqn_status(episode, last_10_scores, episode_option_executions, last_10_durations)

        return per_episode_scores, per_episode_durations

def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
    parser.add_argument("--seed", type=int, help="Random seed for this run (default=0)", default=0)
    parser.add_argument("--train_episodes", type=int, help="# episodes", default=750)
    parser.add_argument("--train_steps", type=int, help="# steps", default=2000)
    parser.add_argument("--test_episodes", type=int, help="# eval episodes", default=750)
    parser.add_argument("--test_steps", type=int, help="# steps per eval episode", default=2000)
    parser.add_argument("--subgoal_reward", type=float, help="SkillChaining subgoal reward", default=0.)
    parser.add_argument("--dense_reward", type=bool, help="Use dense/sparse rewards", default=False)
    parser.add_argument("--lr_a", type=float, help="DDPG Actor learning rate", default=1e-4)
    parser.add_argument("--lr_c", type=float, help="DDPG Critic learning rate", default=1e-3)
    parser.add_argument("--ddpg_batch_size", type=int, help="DDPG Batch Size", default=64)
    parser.add_argument("--render", type=bool, help="Render the mdp env", default=False)
    parser.add_argument("--option_timeout", type=bool, help="Whether option times out at 200 steps", default=False)
    parser.add_argument("--generate_plots", type=bool, help="Whether or not to generate plots", default=False)
    parser.add_argument("--tensor_log", type=bool, help="Enable tensorboard logging", default=False)
    parser.add_argument("--max_num_options", type=int, help="Max number of options we can learn", default=5)
    parser.add_argument("--num_subgoal_hits", type=int, help="Number of subgoal hits to learn an option", default=3)
    parser.add_argument("--buffer_len", type=int, help="buffer size used by option to create init sets", default=20)
    parser.add_argument("--classifier_type", type=str, help="ocsvm/elliptic for option initiation clf", default="ocsvm")
    parser.add_argument("--use_smdp_update", type=bool, help="sparse/SMDP update for option policy", default=False)
    args = parser.parse_args()

    train_env_1 = PortablePointMazeMDP(args.seed, train_mode=True, test_mode=False, dense_reward=args.dense_reward, render=args.render)
    train_env_2 = PortablePointMazeMDP(args.seed, train_mode=False, test_mode=False, dense_reward=args.dense_reward, render=args.render)
    test_env = PortablePointMazeMDP(args.seed, train_mode=False, test_mode=True, dense_reward=args.dense_reward, render=args.render)

    # Create folders for saving various things
    logdir = create_log_dir(args.experiment_name)
    create_log_dir("saved_runs")
    create_log_dir("value_function_plots")
    create_log_dir("initiation_set_plots")
    create_log_dir("value_function_plots/{}".format(args.experiment_name))
    create_log_dir("initiation_set_plots/{}".format(args.experiment_name))

    print("Training skill chaining agent with subgoal reward {} and buffer_len = {}".format(args.subgoal_reward,
                                                                                            args.buffer_len))

    portable_agent = PortableSkillChainingAgent(train_mdp_1=train_env_1, train_mdp_2=train_env_2, test_mdp=test_env,
                                                train_episodes=args.train_episodes,
                                                train_steps=args.train_steps,
                                                test_episodes=args.test_episodes, test_steps=args.test_steps,
                                                lr_actor=args.lr_a, lr_critic=args.lr_c, ddpg_batch_size=args.ddpg_batch_size,
                                                seed=args.seed, experiment_name=args.experiment_name,
                                                subgoal_reward=args.subgoal_reward, buffer_length=args.buffer_len,
                                                log_dir=logdir, num_subgoal_hits_required=args.num_subgoal_hits,
                                                enable_option_timeout=args.option_timeout, generate_plots=args.generate_plots,
                                                use_full_smdp_update=args.use_smdp_update,
                                                tensor_log=args.tensor_log, device=args.device,
                                                num_training_episodes=args.train_episodes, num_training_steps=args.train_steps,
                                                num_test_episodes=args.test_episodes, num_test_steps=args.test_steps)

    training_scores, training_durations = portable_agent.train()
    portable_agent.dsc_agent_1.save_all_scores(pretrained=False, scores=training_scores, durations=training_durations)

    portable_agent.create_transfer_agent()

    eval_scores, eval_durations = portable_agent.evaluate()
    portable_agent.transfer_dsc_agent.save_all_scores(pretrained=True, scores=eval_scores, durations=eval_durations)
    portable_agent.transfer_dsc_agent.perform_experiments()
