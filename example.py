"""
Two examples for inferring the graph structure underlying continuous data and discrete data.
In each of the examples, the data (and all the belonging parameters) could either be simulated
or read from an external file.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import trilearn.distributions.g_inv_wishart as gwish
import trilearn.graph.decomposable
import trilearn.graph.graph
import trilearn.smc as smc


## Continuous data
# Load data from file
df = pd.read_csv("sample_data/dataset_p15.csv", header=None)
graph_traj = smc.gen_pgibbs_ggm_trajectory(dataframe=df, n_particles=50, n_samples=10)
graph_traj.autocorrelation_size().plot()
plt.show()
graph_traj.likelihood().plot()
plt.show()
graph_traj.plot_heatmap()
plt.show()

## Discrete data
# Load data from file
df = pd.read_csv("sample_data/czech_autoworkers.csv", header=[0, 1]) # read labels and number of levels from row 0 and 1
graph_traj = smc.gen_pgibbs_loglin_trajectory(dataframe=df, n_particles=N, n_samples=T)
graph_traj.autocorrelation_size().plot()
plt.show()
graph_traj.likelihood().plot()
plt.show()
graph_traj.plot_heatmap()
plt.show()


# Sample synthetic data
p = 5
n_datasamples = 1000
# sample graph
graph = trilearn.graph.decomposable.sample_decomposable_graph(p)
#glib.plot_adjmat(graph)
#plt.show()
# sample precision matrix
omega = gwish.sample(graph, p, np.matrix(np.identity(p)))
# sample normal data
x = np.random.multivariate_normal(np.zeros(p), omega.I, n_datasamples)
# generate pgibbs trajectory
graph_traj = smc.particle_gibbs_ggm(X=x, n_particles=50, n_samples=10, debug=True)
graph_traj.autocorrelation_size().plot()
plt.show()
graph_traj.likelihood().plot()
plt.show()
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