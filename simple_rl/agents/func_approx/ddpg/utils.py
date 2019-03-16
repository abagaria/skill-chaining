import os
import torch
import pickle


def save_model(ddpg_agent, episode_number, best=True):
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

    if not os.path.exists("saved_runs"):
        os.makedirs("saved_runs")

    prefix = "best_" if best else "final_"
    name = prefix + ddpg_agent.name
    torch.save(actor_state, "saved_runs/{}_actor.pkl".format(name))
    torch.save(critic_state, "saved_runs/{}_critic.pkl".format(name))
    torch.save(target_actor_state, "saved_runs/{}_target_actor.pkl".format(name))
    torch.save(target_critic_state, "saved_runs/{}_target_critic.pkl".format(name))

    with open("saved_runs/{}_replay_buffer.pkl".format(name), "wb") as f:
        pickle.dump(ddpg_agent.replay_buffer, f)

def load_model(ddpg_agent, best=True):
    prefix = "best_" if best else "final_"
    name = prefix + ddpg_agent.name

    actor_state = torch.load("saved_runs/{}_actor.pkl".format(name))
    critic_state = torch.load("saved_runs/{}_critic.pkl".format(name))
    target_actor_state = torch.load("saved_runs/{}_target_actor.pkl".format(name))
    target_critic_state = torch.load("saved_runs/{}_target_critic.pkl".format(name))

    ddpg_agent.actor.load_state_dict(actor_state["state_dict"])
    ddpg_agent.actor_optimizer.load_state_dict(actor_state["optimizer"])
    ddpg_agent.critic.load_state_dict(critic_state["state_dict"])
    ddpg_agent.critic_optimizer.load_state_dict(critic_state["optimizer"])
    ddpg_agent.target_actor.load_state_dict(target_actor_state["state_dict"])
    ddpg_agent.target_critic.load_state_dict(target_critic_state["state_dict"])

    assert actor_state["epoch"] == critic_state["epoch"] == target_actor_state["epoch"] == target_critic_state["epoch"]
    episode = actor_state["epoch"]

    with open("saved_runs/{}_replay_buffer.pkl".format(name), "rb") as f:
        ddpg_agent.replay_buffer = pickle.load(f)

    return episode, ddpg_agent


def compute_gradient_norm(model):
    total_norm = 0.
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm