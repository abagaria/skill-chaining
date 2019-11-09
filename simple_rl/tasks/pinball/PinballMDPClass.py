# Python Imports.
from __future__ import print_function
import copy
import numpy as np

# Other imports.
from rlpy.Domains.Pinball import Pinball
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.pinball.PinballStateClass import PinballState
import pdb
import numpy as np

class PinballMDP(MDP):
    """ Class for pinball domain. """

    def __init__(self, noise=0., episode_length=1000, reward_scale=1000., render=False):
        self.domain = Pinball(noise=noise, episodeCap=episode_length)
        self.render = render
        self.reward_scale = reward_scale

        # Each observation from domain.step(action) is a tuple of the form reward, next_state, is_term, possible_actions
        # s0 returns initial state, is_terminal, possible_actions
        init_observation = self.domain.s0()
        init_state = tuple(init_observation[0])

        actions = self.domain.actions

        MDP.__init__(self, actions, self._transition_func, self._reward_func, init_state=PinballState(*init_state))

    def _reward_func(self, state, action, option_idx=None):
        """
        Args:
            state (PinballState)
            action (int): number between 0 and 4 inclusive
            option_idx (int): was the action selected based on an option policy

        Returns:
            next_state (PinballState)
        """
        assert self.is_primitive_action(action), "Can only implement primitive actions to the MDP"
        reward, obs, done, possible_actions = self.domain.step(action)

        if self.render:
            self.domain.showDomain(action)

        self.next_state = PinballState(*tuple(obs), is_terminal=done)

        assert done == self.is_goal_state(self.next_state), "done = {}, s' = {} should match".format(done, self.next_state)

        return np.clip(reward, -1., 10.)


    def _transition_func(self, state, action):
        return self.next_state

    def execute_agent_action(self, action, option_idx=None):
        """
        Args:
            action (str)
            option_idx (int): given if action came from an option policy

        Returns:
            (tuple: <float,State>): reward, State

        Summary:
            Core method of all of simple_rl. Facilitates interaction
            between the MDP and an agent.
        """
        reward = self.reward_func(self.cur_state, action, option_idx)
        next_state = self.transition_func(self.cur_state, action)
        self.cur_state = next_state

        return reward, next_state

    def reset(self):
        init_observation = self.domain.s0()
        init_state = tuple(init_observation[0])
        self.init_state = PinballState(*init_state)
        cur_state = copy.deepcopy(self.init_state)
        self.set_current_state(cur_state)

    def set_current_state(self, new_state):
        self.cur_state = new_state
        self.domain.state = new_state.features()

    def is_goal_state(self, state):
        """
        We will pass a reference to the PinballModel function that indicates
        when the ball hits its target.
        Returns:
            is_goal (bool)
        """
        target_pos = np.array(self.domain.environment.target_pos)
        target_rad = self.domain.environment.target_rad
        return np.linalg.norm(state.get_position() - target_pos) < target_rad

    @staticmethod
    def is_primitive_action(action):
        assert action >= 0, "Action cannot be negative {}".format(action)
        return action < 5

    def __str__(self):
        return "RlPy_Pinball_Domain"

    def __repr__(self):
        return str(self)

