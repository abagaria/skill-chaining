import random
import pickle
import numpy as np
import itertools
import glob
import os
import ipdb

from simple_rl.mdp.StateClass import State
from simple_rl.agents.func_approx.dsc.SalientEventClass import SalientEvent
from simple_rl.agents.func_approx.dsc.ChainClass import SkillChain

# ---------––––––––---------––––––––---------––––––––
# Eval utils
# ---------––––––––---------––––––––---------––––––––

def create_test_event(dsg_agent, goal_state):
    assert isinstance(goal_state, (State, np.ndarray))

    events = dsg_agent.mdp.get_all_target_events_ever() + [dsg_agent.mdp.get_start_state_salient_event()]
    event_indices = [event.event_idx for event in events]
    event_idx = max(event_indices) + 1
    test_event = SalientEvent(target_state=goal_state, event_idx=event_idx)

    known_events = [event for event in events if event.is_subset(test_event)]

    if len(known_events) > 0:
        sorted_known_events = sorted(known_events, key=lambda e: len(e.trigger_points), reverse=True)
        return sorted_known_events[0]

    # # Is there an option that is a subset of the test event?
    # known_options = [option for option in dsg_agent.planning_agent.plan_graph.option_nodes if
    #                  SkillChain.should_exist_edge_from_option_to_event(option, test_event)]
    #
    # if len(known_options) > 0:
    #     ipdb.set_trace()
    #     sorted_known_options = sorted(known_options, key=lambda o: len(o.effect_set), reverse=True)
    #     return sorted_known_options[0]

    # Add target position to event's trigger points to enable VF based distance calculation
    s0 = np.zeros((dsg_agent.mdp.get_state_space_size(),))
    s0[:2] = test_event.get_target_position()
    test_event.trigger_points.append(s0)

    return test_event


def success_curve(dsg_agent, goal_state, num_episodes, num_steps, start_state=None):

    print("*" * 80)
    print(f"Generating success curve from {start_state} -> {goal_state}")
    print("*" * 80)

    test_event = create_test_event(dsg_agent, goal_state)
    successes = dsg_agent.dsg_run_loop(num_episodes, num_steps, start_state, test_event, eval_mode=True)
    return successes


def learning_curve(dsg_agent, start_states, goal_states, num_episodes, num_steps):

    scores = []
    for start_state, goal_state in zip(start_states, goal_states):
        successes = success_curve(dsg_agent, goal_state, num_episodes, num_steps, start_state)
        scores.append(successes)
    return scores


def generate_learning_curve(dsg_agent, num_episodes, num_steps, num_goals, randomize_start_states=False):

    sampling_func = dsg_agent.mdp.sample_random_state
    goal_states = [sampling_func()[:2] for _ in range(num_goals)]
    start_states = [sampling_func()[:2] for _ in range(num_goals)] if randomize_start_states else [None] * num_goals

    scores = learning_curve(dsg_agent, start_states, goal_states, num_episodes, num_steps)

    file_name = f"dsg_{num_goals}_random_starts_{randomize_start_states}_scores_{dsg_agent.seed}.pkl"
    print("*" * 80); print(f"Saving DSG scores to {dsg_agent.experiment_name}/{file_name}"); print("*" * 80)
    with open(f"{dsg_agent.experiment_name}/{file_name}", "wb+") as f:
        pickle.dump(scores, f)

    # Save the goals states
    goals_file_name = f"dsg_{num_goals}_random_starts_{randomize_start_states}_goals_{dsg_agent.seed}.pkl"
    print("*" * 80)
    print(f"Saving DSG goals to {dsg_agent.experiment_name}/{goals_file_name}")
    print("*" * 80)
    with open(f"{dsg_agent.experiment_name}/{goals_file_name}", "wb+") as f:
        pickle.dump(goal_states, f)

    if randomize_start_states:
        starts_file_name = f"dsg_{num_goals}_random_starts_{randomize_start_states}_start_states_{dsg_agent.seed}.pkl"
        print("*" * 80)
        print(f"Saving DSG start states to {dsg_agent.experiment_name}/{starts_file_name}")
        print("*" * 80)
        with open(f"{dsg_agent.experiment_name}/{starts_file_name}", "wb+") as f:
            pickle.dump(start_states, f)

    return scores


# ---------––––––––---------––––––––---------––––––––
# Saving and loading utils
# ---------––––––––---------––––––––---------––––––––

def save_options_and_events(options, events, experiment_name, seed):

    # Ensure the log directories exist, else create it
    dir_path = os.path.join(os.path.join(os.getcwd(), experiment_name), f"{seed}")
    option_dir_path = os.path.join(dir_path, "options")
    event_dir_path = os.path.join(dir_path, "events")
    os.makedirs(option_dir_path, exist_ok=True)
    os.makedirs(event_dir_path, exist_ok=True)

    # Save all the given options and salient events
    for option in options:
        with open(f"{option_dir_path}/{option.name}.pkl", "wb+") as f:
            pickle.dump(option, f)
    for event in events:
        with open(f"{event_dir_path}/event_{event}.pkl", "wb+") as f:
            pickle.dump(event, f)

    print(f"Done saving {options}")
    print(f"Done saving {events}")


def load_options_and_events(experiment_name, seed):
    """ Given a directory path, return all the options pickled in that directory. """
    dir_path = os.path.join(os.path.join(os.getcwd(), experiment_name), f"{seed}")
    option_dir_path = os.path.join(dir_path, "options")
    event_dir_path = os.path.join(dir_path, "events")

    def _load(path_to_files):
        """ Assuming only relevant object pickle files are in the dir. """
        loaded_objects = []
        for file_name in glob.glob(path_to_files):
            with open(file_name, "rb") as f:
                obj = pickle.load(f)
            print(f"Loaded {obj} from {file_name}")
            loaded_objects.append(obj)
        return loaded_objects

    # Glob pattern for options and events
    option_path = os.path.join(option_dir_path, "*.pkl")
    event_path = os.path.join(event_dir_path, "*.pkl")

    # Final objects to be returned
    options = _load(option_path)
    events = _load(event_path)

    return options, events
