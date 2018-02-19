"""
Two examples for inferring the graph structure underlying continuous data and discrete data.
In each of the examples, the data (and all the belonging parameters) could either be simulated
or read from an external file.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import trilearn.distributions.discrete_dec_log_linear as loglin
import trilearn.distributions.g_inv_wishart as gwish
import trilearn.graph.decomposable
import trilearn.graph.graph
import trilearn.graph.graph as glib
import trilearn.smc as smc


## Continuous data
# Sample synthetic data
p = 5
n = 1000
# sample graph
graph = trilearn.graph.decomposable.sample_decomposable_graph(p)
#glib.plot_adjmat(graph)
#plt.show()
# sample precision matrix
omega = gwish.sample(graph, p, np.matrix(np.identity(p)))
# sample normal data
x = np.matrix(np.random.multivariate_normal(np.zeros(p), omega.I, n))
# generate pgibbs trajectory
graph_traj = smc.particle_gibbs_ggm(X=x, alpha=0.5, beta=0.5, n_particles=50,
                                    traj_length=10, D=np.identity(p), delta=1.0,
                                    radius=p, debug=True)
graph_traj.autocorrelation_size().plot()
plt.show()
graph_traj.likelihood().plot()
plt.show()
graph_traj.plot_heatmap()
plt.show()

# Load data from file
df = pd.read_csv("sample_data/carvalho.csv", header=None)
p = df.shape[1]
x = np.matrix(df.get_values())
graph_traj = smc.particle_gibbs_ggm(X=x, alpha=0.5, beta=0.5, n_particles=50, traj_length=10,
                                    D=np.identity(p), delta=1.0,
                                    radius=p, debug=True)

print graph_traj.autocorrelation_size()
graph_traj.autocorrelation_size().plot()
plt.show()
graph_traj.likelihood().plot()
plt.show()
graph_traj.plot_heatmap()
plt.show()

## Discrete data

# Load data from file
df = pd.read_csv("sample_data/czech_autoworkers.csv", sep=',', header=[0, 1]) # read labels and number of levels from row 0 and 1
p = df.shape[1]
T = 10
N = 50
n_levels = [int(a[1]) for a in list(df.columns)]
levels = np.array([range(l) for l in n_levels])
x = df.values.astype(int)
graph_traj = smc.gen_pgibbs_loglin_trajectory(X=x,
                                              levels=levels,
                                              trajectory_length=T,
                                              n_particles=N)
graph_traj.plot_heatmap()
plt.show()

# # Generate discrete data
# p = 5
# n = 1000
# # sample graph
# graph = trilearn.graph.decomposable.sample_decomposable_graph(p)
# glib.plot_adjmat(graph)
# plt.show()
# levels = np.array([range(2) for l in range(p)])
# pseudo_obs = 1.0
# parameters = loglin.gen_globally_markov_distribution(graph, pseudo_obs, levels) # TODO: There is a bug here
