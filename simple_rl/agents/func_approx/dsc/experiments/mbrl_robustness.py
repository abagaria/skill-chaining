import numpy as np
from copy import deepcopy
from collections import deque
from simple_rl.agents.func_approx.dsc.dynamics.mpc import MPC
from simple_rl.agents.func_approx.dsc.OptionClass import Option
from simple_rl.tasks.ant_reacher.AntReacherMDPClass import AntReacherMDP


class ComposeModelBasedOptions(object):
    def __init__(self, device, initiation_period=5, use_warmup_phase=True):
        self.device = device
        self.initiation_period = initiation_period
        self.use_warmup_phase = use_warmup_phase

        self.mdp = AntReacherMDP(task_agnostic=False, goal_state=np.array((8, 0)), seed=0)
        self.init_salient_event = self.mdp.get_start_state_salient_event()
        self.target_salient_event = self.mdp.get_original_target_events()[0]

        self.goal_option = Option(self.mdp, "goal_option", None, **self.get_common_option_kwargs())
        self.init_option = Option(self.mdp, "init_option", None, **self.get_common_option_kwargs())

    def act(self, state):
        if self.goal_option.is_init_true(state):
            return self.goal_option
        return self.init_option

    def run_loop(self, num_episodes=1000, num_steps=150):
        per_episode_scores = []
        per_episode_durations = []
        last_10_scores = deque(maxlen=10)
        last_10_durations = deque(maxlen=10)

        for episode in range(num_episodes):

            score = 0
            self.mdp.reset()
            state = deepcopy(self.mdp.cur_state)

            for step_number in range(num_steps):

                selected_option = self.act(state)
                selected_option.execute_option_in_mdp(self.mdp,
                                                      episode,
                                                      step_number,
                                                      self.mdp.goal_state,
                                                      eval_mode=False,
                                                      goal_salient_event=self.target_salient_event)
                state = deepcopy(self.mdp.cur_state)

                if state.is_terminal():
                    break

            last_10_scores.append(score)
            last_10_durations.append(step_number)
            per_episode_scores.append(score)
            per_episode_durations.append(step_number)

            self.log_status(episode, last_10_scores, last_10_durations)

        return per_episode_scores, per_episode_durations

    def log_status(self, episode, last_10_scores, last_10_durations):
        # Print scores to terminal

        # Plot the initiation classifier for the goal-option

        #
        pass

    def get_common_option_kwargs(self):
        kwargs = {"overall_mdp": self.mdp, "lr_actor": 1e-3, "lr_critic": 1e-4, "ddpg_batch_size": 64,
                  "buffer_len": 100, "dense_reward": True, "timeout": 100, "initiation_period": self.initiation_period,
                  "max_num_children": 1, "init_salient_event": self.init_salient_event,
                  "target_salient_event": self.target_salient_event, "use_warmup_phase": self.use_warmup_phase,
                  "solver_type": "mpc", "use_her": False, "device": self.device}
        return kwargs
