import ipdb
import pickle
import random
import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from megaman.geometry import Geometry
from megaman.embedding import SpectralEmbedding

radius = 10
adjacency_method = 'cyflann'
# higher is better:
adjacency_kwds = {'radius': radius} # ignore distances above this radius
affinity_method = 'gaussian'
# higher is faster:
affinity_kwds = {'radius': radius} # A = exp(-||x - y||/radius^2)
laplacian_method = 'geometric'
# lower is better:
laplacian_kwds = {'scaling_epps': radius} # scaling ensures convergence to Laplace-Beltrami operator

geom  = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds,
                 affinity_method=affinity_method, affinity_kwds=affinity_kwds,
                 laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)

spectral = SpectralEmbedding(n_components=1, eigen_solver='arpack', geom=geom)

# from sklearn.manifold import SpectralEmbedding
# spectral = SpectralEmbedding(n_components=1)

def plot_embeddings(states, values, experiment_name=""):
    min_state, max_state = states[np.argmin(values)], states[np.argmax(values)]
    plt.scatter(states[:, 0], states[:, 1], c=values, cmap="Blues")
    plt.colorbar()
    plt.scatter([min_state[0]], [min_state[1]], c="black", s=[300])
    plt.scatter([max_state[0]], [max_state[1]], c="red", s=[300])

    if not os.path.isdir("initiation_set_plots/{}".format(experiment_name)):
        os.makedirs("initiation_set_plots/{}".format(experiment_name))

    plt.suptitle("Embeddings of {} subsampled ant-maze transitions".format(len(states)))
    plt.savefig("initiation_set_plots/{}/spectral_embeddings_plot.png".format(experiment_name))
    plt.close()

# TODO:
# 1. Run on N=1_000_000 points and let it run for 20 mins
# 2. Check paper (https://arxiv.org/pdf/1603.02763.pdf) to see what radius means
# 3. Check source code if unclear
# 4. Contact authors if can't figure it out

with open("d4rl_transitions.pkl", "rb") as t_file, open("d4rl_embbedings.pkl", "wb") as e_file:
    transitions = pickle.load(t_file)
    transitions = random.sample(transitions, 30000)
    print("read", len(transitions), "transitions")
    states = torch.stack([torch.from_numpy(trans[0]).float() for trans in transitions])
    values = spectral.fit_transform(states)
    pickle.dump(values, e_file)
    plot_embeddings(states.numpy(), values.flatten(), experiment_name="spectral_test")
