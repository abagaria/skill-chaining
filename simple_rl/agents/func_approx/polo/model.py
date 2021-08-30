import torch
import torch.nn as nn
from torch.nn import init


class ValueFunctionModel(nn.Module):
    def __init__(self, state_size, device, hidden_size=16, beta=1.):
        super(ValueFunctionModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.prior = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        # Random orthogonal initialization
        for p in self.modules():
            if isinstance(p, (nn.Linear, nn.Conv2d)):
                init.orthogonal_(p.weight, beta)

        # No back-prop through the prior network
        for p in self.prior.parameters():
            p.requires_grad = False

        self.beta = beta
        self.device = device
        self.to(device)

    def forward(self, states):
        with torch.no_grad():
            prior = self.prior(states)
        return prior.detach() + self.model(states)

    def predict(self, states):
        self.model.eval()
        with torch.no_grad():
            values = self(states)
        self.model.train()
        return values
