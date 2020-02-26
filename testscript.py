import numpy as np
from dm_control import suite
from dm_control import viewer
from dm_control.suite.wrappers import pixels
import dmc2gym
from simple_rl.tasks.gym.wrappers import *
#
# from simple_rl.mdp.MDPClass import MDP
# from simple_rl.tasks.dmcontrol_reacher.DMControlReacherStateClass import DMControlReacherState

env = dmc2gym.make(domain_name="reacher", task_name="hard", from_pixels=True, height=84, width=84, visualize_reward=False)

print(env.reset().shape)

env = FrameStack(env, num_stack=4)

print(env.reset().shape)

# for i in range(50):
