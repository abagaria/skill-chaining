import random
import pickle
import numpy as np
import itertools

from simple_rl.agents.func_approx.dsc.DeepSkillGraphAgentClass import DeepSkillGraphAgent

# TODO: Eventually all of this code needs to be moved to MDPPlotterClass
def sample_state_inside_graph(dsg_agent):
    """

    Args:
        dsg_agent (DeepSkillGraphAgent)

    Returns:

    """
    def flatten(l):
        return list(itertools.chain.from_iterable(l))

    options = dsg_agent.planning_agent.get_options_in_known_part_of_the_graph()
    positive_examples = [option.positive_examples for option in options]

    done = False
    sampled_state = None

    while not done:
        sampled_state = random.choice(flatten(flatten(positive_examples)))
        done = dsg_agent.planning_agent.is_state_inside_graph(sampled_state)

    assert sampled_state is not None

    return sampled_state


def sample_state_outside_graph(dsg_agent, mdp):
    num_tries = 0
    done = False
    sampled_state = None
    while not done and num_tries < 200:
        sampled_state = generate_test_states(mdp, num_states=1)[0]
        done = not dsg_agent.planning_agent.is_state_inside_graph(sampled_state)
    assert sampled_state is not None
    return sampled_state


def learning_curve_inside_graph(agent, mdp, episodes, episode_interval, randomize_start_states=False):
    start_states = [sample_state_inside_graph(dsg_agent=agent) for _ in range(5)]
    goal_states = [sample_state_inside_graph(dsg_agent=agent) for _ in range(5)]

    if randomize_start_states:
        all_runs = learning_curve_with_start_states(agent, start_states, goal_states, episodes, episode_interval)
        return all_runs

    all_runs = learning_curve_without_random_start_states(agent, goal_states, episodes, episode_interval)
    return all_runs


def learning_curve_outside_graph(agent, mdp, episodes, episode_interval, randomize_start_states=False):
    start_states = [sample_state_inside_graph(dsg_agent=agent) for _ in range(5)]
    goal_states = [sample_state_outside_graph(dsg_agent=agent, mdp=mdp) for _ in range(5)]

    if randomize_start_states:
        all_runs = learning_curve_with_start_states(agent, start_states, goal_states, episodes, episode_interval)
        return all_runs

    all_runs = learning_curve_without_random_start_states(agent, goal_states, episodes, episode_interval)
    return all_runs


def learning_curve_with_start_states(agent, start_states, goal_states, episodes, episode_interval):
    all_runs = []
    for start_state, goal_state in zip(start_states, goal_states):
        single_run = success_curve(agent,
                                   start_state,
                                   goal_state,
                                   episodes,
                                   episode_interval)
        all_runs.append(single_run)
    return all_runs


def learning_curve_without_random_start_states(agent, goal_states, episodes, episode_interval):
    all_runs = []
    for goal_state in goal_states:
        single_run = success_curve(agent,
                                   None,
                                   goal_state,
                                   episodes,
                                   episode_interval)
        all_runs.append(single_run)
    return all_runs
