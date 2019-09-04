import torch
import torch.nn
import torch.nn.functional as F


class NN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, n_hid):
        super(NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hid = n_hid

        self.fc1 = torch.nn.Linear(in_dim, n_hid)
        self.fc2 = torch.nn.Linear(n_hid, n_hid)
        self.fc3 = torch.nn.Linear(n_hid, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return y


class RND:
    def __init__(self, in_dim, out_dim, n_hid, device):
        self.device = device
        self.target = NN(in_dim, out_dim, n_hid).to(self.device)
        self.model = NN(in_dim, out_dim, n_hid).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)

    def get_reward(self, x):
        y_true = self.target(x).detach()
        y_pred = self.model(x)
        reward = F.mse_loss(y_pred, y_true)  #torch.pow(y_pred - y_true, 2).sum()
        return reward

    def update(self, Ri):
        Ri.sum().backward()
        self.optimizer.step()

    def get_single_reward(self, state):
        """ state is a numpy array of features. """
        x = torch.from_numpy(state).float().to(self.device)
        with torch.no_grad():
            y_true = self.target(x).detach()
            y_pred = self.model(x)
        reward = F.mse_loss(y_pred, y_true)
        clipped_reward = reward.detach().clamp(-1., 1.).item()
        return clipped_reward