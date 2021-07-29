import pfrl
import torch
from torch import nn
from pfrl.policies import SoftmaxCategoricalHead


def lecun_init(layer, gain=1):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        pfrl.initializers.init_lecun_normal(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    else:
        pfrl.initializers.init_lecun_normal(layer.weight_ih_l0, gain)
        pfrl.initializers.init_lecun_normal(layer.weight_hh_l0, gain)
        nn.init.zeros_(layer.bias_ih_l0)
        nn.init.zeros_(layer.bias_hh_l0)
    return layer


def ortho_init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)


def conv_model(obs_n_channels, n_actions):
    return nn.Sequential(
        lecun_init(nn.Conv2d(obs_n_channels, 32, 8, stride=4)),
        nn.ReLU(),
        lecun_init(nn.Conv2d(32, 64, 4, stride=2)),
        nn.ReLU(),
        lecun_init(nn.Conv2d(64, 64, 3, stride=1)),
        nn.ReLU(),
        nn.Flatten(),
        lecun_init(nn.Linear(3136, 512)),
        nn.ReLU(),
        pfrl.nn.Branched(
            nn.Sequential(
                lecun_init(nn.Linear(512, n_actions), 1e-2),
                SoftmaxCategoricalHead(),
            ),
            lecun_init(nn.Linear(512, 1)),
        ),
    )


def fc_model(obs_size, action_size):
    policy = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, action_size),
        pfrl.policies.GaussianHeadWithStateIndependentCovariance(
            action_size=action_size,
            var_type="diagonal",
            var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
            var_param_init=0,  # log std = 0 => std = 1
        ),
    )

    vf = torch.nn.Sequential(
        nn.Linear(obs_size, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 1),
    )

    ortho_init(policy[0], gain=1)
    ortho_init(policy[2], gain=1)
    ortho_init(policy[4], gain=1e-2)
    ortho_init(vf[0], gain=1)
    ortho_init(vf[2], gain=1)
    ortho_init(vf[4], gain=1)

    return pfrl.nn.Branched(policy, vf)
