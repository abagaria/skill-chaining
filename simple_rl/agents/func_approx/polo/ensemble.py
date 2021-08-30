import ipdb
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim import Adam
from collections import deque
from torch.utils.data import DataLoader
from simple_rl.agents.func_approx.polo.model import ValueFunctionModel


class ValueFunctionEnsemble:
    def __init__(self, state_size, ensemble_size, device, lam=0.01, batch_size=32):
        self.device = device
        self.batch_size = batch_size
        self.ensemble_size = ensemble_size
        self.members = [ValueFunctionModel(state_size, device) for _ in range(ensemble_size)]
        self.ensemble_buffers = [deque(maxlen=int(1e5)) for _ in range(ensemble_size)]
        self.optimizers = [Adam(x.model.parameters(), weight_decay=lam) for x in self.members]

    def __call__(self, input_data):
        if isinstance(input_data, np.ndarray):
            input_data = torch.FloatTensor(input_data).to(self.device)

        predictions = [member.predict(input_data).cpu().numpy() for member in self.members]
        predictions = np.array(predictions).squeeze()

        return predictions.mean(axis=0), predictions.std(axis=0) ** 2

    def sample(self, member_idx):
        state_buffer = self.ensemble_buffers[member_idx]
        sampled_states = random.sample(state_buffer, self.batch_size)
        sampled_states = np.array(sampled_states)
        return torch.FloatTensor(sampled_states).to(self.device)

    def add_data(self, state):
        """ Add a single state - value pair to the bootstrap ensemble data buffers."""
        for buffer in self.ensemble_buffers:
            if random.random() >= 0.5:
                buffer.append(state)
