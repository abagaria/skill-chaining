import random
import pickle
import numpy as np
import itertools
import glob
import os

from simple_rl.agents.func_approx.dsc.DeepSkillGraphAgentClass import DeepSkillGraphAgent


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


def generate_test_states(mdp, num_states=10):
    generated_states = []
    for i in range(num_states):
        goal_position = mdp.sample_random_state()[:2]
        generated_states.append(goal_position)
    return generated_states


def success_curve(dsg_agent, start_state, goal_state, episodes, episode_interval):
    success_rates_over_time = []
    for episode in range(episodes):
        success_rate = dsg_agent.planning_agent.measure_success(goal_state=goal_state,
                                                                start_state=start_state,
                                                                starting_episode=episode,
                                                                num_episodes=episode_interval)

        print("*" * 80)
        print(f"[{goal_state}]: Episode {episode}: Success Rate: {success_rate}")
        print("*" * 80)

        success_rates_over_time.append(success_rate)
    return success_rates_over_time


def learning_curve(agent, mdp, episodes, episode_interval, randomize_start_states=False):
    start_states = generate_test_states(mdp)
    goal_states = generate_test_states(mdp)
    all_runs = []
    for start_state in start_states:
        for goal_state in goal_states:
            start_state = start_state if randomize_start_states else None
            single_run = success_curve(agent,
                                       start_state,
                                       goal_state,
                                       episodes,
                                       episode_interval)
            all_runs.append(single_run)
    return all_runs


def learning_curve_inside_graph(agent, mdp, episodes, episode_interval, randomize_start_states=False,
                                experiment_name="", num_runs=5):
    start_states = [sample_state_inside_graph(dsg_agent=agent) for _ in range(num_runs)]
    goal_states = [sample_state_inside_graph(dsg_agent=agent) for _ in range(num_runs)]

    if randomize_start_states:
        with open(f"{experiment_name}/lc_randomized_start_states_{randomize_start_states}_start_states.pkl",
                  "wb+") as f:
            pickle.dump(start_states, f)

    with open(f"{experiment_name}/lc_randomized_start_states_{randomize_start_states}_goal_states.pkl", "wb+") as f:
        pickle.dump(goal_states, f)

    if randomize_start_states:
        all_runs = learning_curve_with_start_states(agent, start_states, goal_states, episodes, episode_interval)
        return all_runs

    all_runs = learning_curve_without_random_start_states(agent, goal_states, episodes, episode_interval)
    return all_runs


def learning_curve_fully_random(agent, mdp, episodes, episode_interval, randomize_start_states=False, experiment_name="", num_runs=5):
    start_states = [mdp.sample_random_state() for _ in range(num_runs)]
    goal_states = [mdp.sample_random_state() for _ in range(num_runs)]

    if randomize_start_states:
        with open(f"{experiment_name}/lc_randomized_start_states_{randomize_start_states}_start_states.pkl",
                  "wb+") as f:
            pickle.dump(start_states, f)

    with open(f"{experiment_name}/lc_randomized_start_states_{randomize_start_states}_goal_states.pkl", "wb+") as f:
        pickle.dump(goal_states, f)

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


def save_options(options, dir_path):
    for option in options:
        with open(f"{dir_path}/{option.name}.pkl", "wb+") as f:
            pickle.dump(option, f)
    print(f"Done saving {options}")


def load_options(dir_path):
    """ Given a directory path, return all the options pickled in that directory. """
    path_name = os.path.join(dir_path, "*.pkl")
    options = []
    for file_name in glob.glob(path_name):
        with open(file_name, "rb") as f:
            option = pickle.load(f)
        print(f"Loaded {option} from {file_name}")
        options.append(option)
    return options
