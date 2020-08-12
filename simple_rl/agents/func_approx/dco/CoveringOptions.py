import ipdb
import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from simple_rl.agents.func_approx.dco.utils import *
from simple_rl.agents.func_approx.dco.model import SpectrumNetwork, LinearNetwork, PositionNetwork
from simple_rl.agents.func_approx.dco.dataset import DCODataset
from simple_rl.agents.func_approx.dco.losses import get_loss_fn
from simple_rl.agents.func_approx.dqn.DQNAgentClass import ReplayBuffer
from simple_rl.agents.func_approx.dsc.SalientEventClass import DCOSalientEvent
from simple_rl.agents.func_approx.dsc.utils import plot_dco_salient_event_comparison


class CoveringOptions(object):
    def __init__(self, state_dim, device, use_xy=False, sub_sample=False,
                 lr=1e-3, beta=2.0, delta=1.0, seed=0, loss_type="var"):
        self.beta = beta
        self.delta = delta
        self.device = device
        self.use_xy = use_xy
        self.loss_type = loss_type
        self.sub_sample = sub_sample

        if self.use_xy:
            self.model = LinearNetwork(state_dim, device=self.device)
        else:
            self.model = SpectrumNetwork(state_dim, device=self.device)

        self.seed = random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # When beta is the lagrange multiplier, it must be a variable in the optimization
        if self.loss_type == "lagrangian":
            self.beta = torch.Tensor([beta]).to(self.device)
            self.beta.requires_grad = True
            params = list(self.model.parameters()) + [self.beta]
            self.optimizer = optim.Adam(params, lr=lr)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.writer = SummaryWriter()

        self.max_f_value_state = None
        self.min_f_value_state = None

        # For debugging
        self.max_states = []
        self.min_states = []
        self.name = "Ant-reacher-full-state"

    def train(self, replay_buffer, n_epochs=1):
        assert isinstance(replay_buffer, ReplayBuffer)

        losses = []
        dataset = DCODataset(replay_buffer, use_xy=self.use_xy, sub_sample=self.sub_sample)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        for epoch in range(n_epochs):

            self.model.train()

            for states1, states2, next_states in tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}"):
                states1 = states1.to(self.device)
                states2 = states2.to(self.device)
                next_states = next_states.to(self.device)

                f_val1 = self.model(states1)
                f_val2 = self.model(states2)
                f_next_val = self.model(next_states)

                loss_fn = get_loss_fn(name=self.loss_type)
                loss = loss_fn(f_s1=f_val1, f_sp=f_next_val,
                               f_s2=f_val2, beta=self.beta, delta=self.delta)

                losses.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self._determine_max_and_min_states(replay_buffer)

        return losses

    def evaluate(self, states, chunk_size=1000):
        assert isinstance(states, np.ndarray)

        if self.use_xy:
            states = states[:, :2]

        # Chunk up the inputs so as to conserve GPU memory
        num_chunks = int(np.ceil(states.shape[0] / chunk_size))

        if num_chunks == 0:
            return 0.

        state_chunks = np.array_split(states, num_chunks, axis=0)
        f_values = np.zeros((states.shape[0],))

        current_idx = 0
        self.model.eval()

        for state_chunk in tqdm(state_chunks, desc="Fetching F-Values"):
            state_chunk = torch.from_numpy(state_chunk).float().to(self.device)
            with torch.no_grad():
                chunk_values = self.model(state_chunk).detach().cpu().numpy()
            current_chunk_size = len(state_chunk)
            f_values[current_idx:current_idx + current_chunk_size] = chunk_values.squeeze(1)
            current_idx += current_chunk_size

        return f_values

    def _determine_max_and_min_states(self, replay_buffer):
        states = np.array([trans.state for trans in replay_buffer.memory])
        f_values = self.evaluate(states)

        self.max_f_value_state = states[np.argmax(f_values)]
        self.min_f_value_state = states[np.argmin(f_values)]

        self.max_states.append(self.max_f_value_state)
        self.min_states.append(self.min_f_value_state)

        print(f"Max State: {self.max_f_value_state[:2]}\t Min State: {self.min_f_value_state[:2]}")
        print(f"Max F-Val: {f_values.max()} \t Min F-Val: {f_values.min()}")

    def get_max_f_value_state(self):
        if self.max_f_value_state is not None:
            return self.max_f_value_state
        raise Warning("DCO model not trained yet")

    def get_min_f_value_state(self):
        if self.min_f_value_state is not None:
            return self.min_f_value_state
        raise Warning("DCO model not trained yet")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, help="Experiment Name")
    parser.add_argument("--device", type=str, help="cpu/cuda:0/cuda:1")
    parser.add_argument("--env", type=str, default="d4rl-ant-maze")
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--loss", type=str, default="var")
    parser.add_argument("--use_xy", action="store_true", default=False)
    parser.add_argument("--collect_new_data", action="store_true", default=False)
    parser.add_argument("--subsample", action="store_true", default=False)
    parser.add_argument("--n_epochs", type=float, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)

    if args.collect_new_data:
        r_buffer = get_random_data(args.env, device)
    else:
        r_buffer = get_saved_transition_data(args.env, device)

    s_dim = 2 if args.use_xy else r_buffer.memory[0][0].shape[0]
    c_option = CoveringOptions(s_dim,
                               device=device,
                               beta=args.beta,
                               delta=args.delta,
                               loss_type=args.loss,
                               sub_sample=args.subsample,
                               use_xy=args.use_xy,
                               )
    l = c_option.train(r_buffer, n_epochs=args.n_epochs)

    low_salient_event = DCOSalientEvent(c_option, 0, is_low=True)
    high_salient_event = DCOSalientEvent(c_option, 1, is_low=False)

    plot_dco_salient_event_comparison(low_salient_event,
                                      high_salient_event,
                                      r_buffer,
                                      0,
                                      False,
                                      False,
                                      args.experiment_name)

    plt.figure()
    plt.loglog(l)
    plt.savefig("dco_dev_loss.png")
    plt.close()
