import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import chordal_learning.distributions.discrete_dec_log_linear as loglin
import chordal_learning.distributions.g_inv_wishart as gwish
import chordal_learning.graph.christmas_tree_algorithm as cta
import chordal_learning.graph.graph as glib
import chordal_learning.smc as smc

# Generated data
p = 5
n = 1000
# sample graph
graph = cta.sample_graph(p)
glib.plot_adjmat(graph)
plt.show()
# sample precision matrix
omega = gwish.sample(graph, p, np.matrix(np.identity(p)))
# sample normal data
x = np.matrix(np.random.multivariate_normal(np.zeros(p), omega.I, n))
# generate pgibbs trajectory
graph_traj = smc.particle_gibbs_ggm(X=x, alpha=0.5, beta=0.5, N=50, traj_length=100, D=np.identity(p), delta=1.0,
                                    radius=p)
graph_traj.plot_heatmap()
plt.show()

# Load data from file
x = np.matrix(np.loadtxt("sample_data/carvalho.csv",
                         delimiter=','))
p = x.shape[1]
graph_traj = smc.particle_gibbs_ggm(X=x, alpha=0.5, beta=0.5, N=50, traj_length=1000, D=np.identity(p), delta=1.0,
                                    radius=p)
    graph_traj.plot_heatmap()
    plt.show()


# Generate discrete data
p = 5
n = 1000
# sample graph
graph = cta.sample_graph(p)
glib.plot_adjmat(graph)
plt.show()
levels = np.array([range(2) for l in range(p)])
pseudo_obs = 1.0
parameters = loglin.gen_globally_markov_distribution(graph, pseudo_obs, levels)

# Load data from file

data_filename = "sample_data/czech_autoworkers.csv"
X = pd.read_csv(data_filename, sep=',').values.astype(int)
levels = np.array([range(l) for l in n_levels])
