import gym
import random
import numpy as np
from simple_rl.mdp.GoalDirectedMDPClass import GoalDirectedMDP
from simple_rl.tasks.montezuma.MRRAMStateClass import MRRAMState


class MontezumaRAMMDP(GoalDirectedMDP):
    def __init__(self, render, seed):
        self.env_name = "MontezumaRevengeNoFrameskip-v4"
        self.env = gym.make(self.env_name)

        self.render = render
        self.seed = seed

        self.env.seed(seed)
        self.reset()

        GoalDirectedMDP.__init__(self,
                                 range(self.env.action_space.n),
                                 self._transition_func,
                                 self._reward_func,
                                 self.init_state,
                                 salient_positions=[],
                                 task_agnostic=True,
                                 dense_reward=False,
                                 goal_tolerance=2)

    def get_current_ram(self):
        return self.env.env.ale.getRAM()

    def _transition_func(self, state, action):
        return self.next_state

    def _reward_func(self, state, action):
        reward = self._step(state, action)
        self.num_lives = self.cur_state.get_num_lives(self.get_current_ram())
        self.skull_pos = self.cur_state.get_skull_position(self.get_current_ram())
        return reward

    def _step(self, state, action):
        _, reward, done, _ = self.env.step(action)
        ram = self.get_current_ram()

        if self.render:
            self.env.render()

        reward = np.sign(reward)  # Reward clipping
        is_dead = self.get_num_lives(ram) < self.num_lives
        skull_direction = int(self.skull_pos > self.get_skull_pos(ram))

        self.next_state = MRRAMState(ram, is_dead, skull_direction, done)

        return reward

    def sparse_gc_reward_function(self, state, goal, info={}, tol=2):
        assert isinstance(state, MRRAMState), type(state)
        assert isinstance(goal, MRRAMState), type(goal)

        def is_close(pos1, pos2):
            return abs(pos1[0] - pos2[0]) <= tol and abs(pos1[1] - pos2[1]) <= tol

        reached = is_close(state.get_position(), goal.get_position())
        reward = +1. if reached else 0.

        return reward, reached

    def reset(self):
        self.env.reset()
        # self.remove_skull()

        for _ in range(4):
            obs, _, _, _ = self.env.step(0)  # no-op to get agent onto ground

        ram = self.env.env.ale.getRAM()
        self.skull_pos = self.get_skull_pos(ram)
        skull_direction = int(self.skull_pos > self.get_skull_pos(ram))
        self.init_state = MRRAMState(ram, skull_direction, False, False)
        self.num_lives = self.init_state.get_num_lives(ram)

        super(MontezumaRAMMDP, self).reset()

    def state_space_size(self):
        return self.init_state.features().shape[0]

    def action_space_size(self):
        return len(self.actions)

    def sample_random_action(self):
        return random.choice(self.actions)

    @staticmethod
    def get_x_y_low_lims():
        return 0, 100

    @staticmethod
    def get_x_y_high_lims():
        return 140, 300

    @staticmethod
    def get_num_lives(ram):
        return int(MRRAMState.getByte(ram, 'ba'))

    @staticmethod
    def get_skull_pos(ram):
        skull_x = int(MRRAMState.getByte(ram, 'af')) + 33
        return skull_x / 100.

    def remove_skull(self):
        print("Setting skull position")
        state_ref = self.env.env.ale.cloneState()
        state = self.env.env.ale.encodeState(state_ref)
        self.env.env.ale.deleteState(state_ref)

        state[431] = 1
        state[351] = 390  # 40

        new_state_ref = self.env.env.ale.decodeState(state)
        self.env.env.ale.restoreState(new_state_ref)
        self.env.env.ale.deleteState(new_state_ref)
        self.execute_agent_action(0)  # NO-OP action to update the RAM state
