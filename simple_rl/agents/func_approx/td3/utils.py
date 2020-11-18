import torch
import copy


def save(td3_agent, filename):
    torch.save(td3_agent.critic.state_dict(), filename + "_critic")
    torch.save(td3_agent.critic_optimizer.state_dict(), filename + "_critic_optimizer")

    torch.save(td3_agent.actor.state_dict(), filename + "_actor")
    torch.save(td3_agent.actor_optimizer.state_dict(), filename + "_actor_optimizer")


def load(td3_agent, filename):
    td3_agent.critic.load_state_dict(torch.load(filename + "_critic"))
    td3_agent.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
    td3_agent.target_critic = copy.deepcopy(td3_agent.critic)

    td3_agent.actor.load_state_dict(torch.load(filename + "_actor"))
    td3_agent.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
    td3_agent.target_actor = copy.deepcopy(td3_agent.actor)
