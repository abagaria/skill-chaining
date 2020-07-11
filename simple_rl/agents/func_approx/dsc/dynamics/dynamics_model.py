import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from simple_rl.agents.func_approx.dsc.dynamics.rollouts import collect_random_rollout

class DynamicsModel(nn.Module):
    def __init__(self, state_size, action_size):
        super(DynamicsModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size + action_size, 500),
            nn.ReLU(),
            nn.Linear(500, state_size)
        )
    
    def forward(self, state, action):
        cat = torch.cat([state, action], dim=1)
        return self.model(cat)

if __name__== "__main__":
    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = DynamicsModel(6, 2)
    model.to(device)
    loss_function = nn.MSELoss().to(device)
    rollout = collect_random_rollout(epochs=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    training_generator = DataLoader(rollout, batch_size=32, shuffle=True, drop_last=True)

    for epoch in range(1, 1000):
        total_loss = 0
        for states, actions, states_p in training_generator:
            states = states.to(device).float()
            actions = actions.to(device).float()
            states_p = states_p.to(device).float()
            
            optimizer.zero_grad()
            states_pred = model.forward(states, actions)
            loss = loss_function(states_p, states_pred)
            loss.backward()
            optimizer.step()
            
            total_loss += loss
        print("Epoch {} total loss {}".format(epoch + 1, total_loss))