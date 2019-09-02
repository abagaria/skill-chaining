from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.lunar_lander.wrappers import *
from simple_rl.tasks.gym.wrappers import *
from simple_rl.tasks.lunar_lander.LunarLanderStateClass import LunarLanderState


class LunarLanderMDP(MDP):
    def __init__(self, render=False, seed=0):
        name = "LunarLander-v2"
        self.env = FrameStack(
            GrayScaleObservation(
                ResizeObservation(
                    PixelObservationWrapper(gym.make(name)),
                    shape=84)
                ),
            num_stack=6)

        self.env.seed(seed)
        self.render = render

        init_obs = self.env.reset()

        MDP.__init__(self, range(self.env.action_space.n), self._transition_func, self._reward_func,
                     init_state=LunarLanderState(init_obs))

    def _to_grayscale(self, image):
        pass

    def _reward_func(self, state, action):
        obs, reward, is_terminal, info = self.env.step(action)

        if self.render:
            self.env.render()

        self.next_state = LunarLanderState(obs, is_terminal=is_terminal)

        return reward

    def _transition_func(self, state, action):
        return self.next_state

    def reset(self):
        self.env.reset()

    def __str__(self):
        return "gym-lunar-lander-pixels"