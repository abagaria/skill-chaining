import os
import torch
import pickle


def save_model(ddpg_agent, episode_number, logdir, best=True, save_ddpg=True):
    if not os.path.exists(os.path.join(logdir, "saved_runs")):
        os.makedirs(os.path.join(logdir, "saved_runs"))

    prefix = "best_" if best else "final_"
    name = prefix + ddpg_agent.name

    if save_ddpg:
        actor_state = {
            "epoch": episode_number,
            "state_dict": ddpg_agent.actor.state_dict(),
            "optimizer": ddpg_agent.actor_optimizer.state_dict()
        }

        critic_state = {
            "epoch": episode_number,
            "state_dict": ddpg_agent.critic.state_dict(),
            "optimizer": ddpg_agent.critic_optimizer.state_dict()
        }

        target_actor_state = {
            "epoch": episode_number,
            "state_dict": ddpg_agent.target_actor.state_dict()
        }

        target_critic_state = {
            "epoch": episode_number,
            "state_dict": ddpg_agent.target_critic.state_dict()
        }

        torch.save(actor_state, os.path.join(logdir, "saved_runs", f"{name}_actor.pkl"))
        torch.save(critic_state, os.path.join(logdir, "saved_runs", f"{name}_critic.pkl"))
        torch.save(target_actor_state, os.path.join(logdir, "saved_runs", f"{name}_target_actor.pkl"))
        torch.save(target_critic_state, os.path.join(logdir, "saved_runs", f"{name}_target_critic.pkl"))

    with open(os.path.join(logdir, "saved_runs", f"{name}_replay_buffer.pkl"), "wb") as f:
        pickle.dump(ddpg_agent.replay_buffer, f)


def load_model(ddpg_agent, logdir, best=True):
    prefix = "best_" if best else "final_"
    name = prefix + ddpg_agent.name

    actor_state = torch.load(os.path.join(logdir, "saved_runs", f"{name}_actor.pkl"))
    critic_state = torch.load(os.path.join(logdir, "saved_runs", f"{name}_critic.pkl"))
    target_actor_state = torch.load(os.path.join(logdir, "saved_runs", f"{name}_target_actor.pkl"))
    target_critic_state = torch.load(os.path.join(logdir, "saved_runs", f"{name}_target_critic.pkl"))

    ddpg_agent.actor.load_state_dict(actor_state["state_dict"])
    ddpg_agent.actor_optimizer.load_state_dict(actor_state["optimizer"])
    ddpg_agent.critic.load_state_dict(critic_state["state_dict"])
    ddpg_agent.critic_optimizer.load_state_dict(critic_state["optimizer"])
    ddpg_agent.target_actor.load_state_dict(target_actor_state["state_dict"])
    ddpg_agent.target_critic.load_state_dict(target_critic_state["state_dict"])

    assert actor_state["epoch"] == critic_state["epoch"] == target_actor_state["epoch"] == target_critic_state["epoch"]
    episode = actor_state["epoch"]

    with open(os.path.join(logdir, "saved_runs", f"{name}_replay_buffer.pkl"), "rb") as f:
        ddpg_agent.replay_buffer = pickle.load(f)

    return episode, ddpg_agent


def compute_gradient_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def create_log_dir(experiment_name):
    path = os.path.join(os.getcwd(), experiment_name)
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)
    return path


def save_all_scores(scores, durations, log_dir, seed):
    print("\rSaving training scores and durations..")
    training_scores_file_name = f"flat_ddpg_training_scores_{seed}"
    training_durations_file_name = f"flat_ddpg_training_durations_{seed}"

    training_scores_file_name = os.path.join(log_dir, training_scores_file_name)
    training_durations_file_name = os.path.join(log_dir, training_durations_file_name)

    with open(training_scores_file_name, "wb+") as _f:
        pickle.dump(scores, _f)
    with open(training_durations_file_name, "wb+") as _f:
        pickle.dump(durations, _f)
