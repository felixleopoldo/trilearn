"""
Examples for inferring the graph structure underlying continuous data and discrete data.
In each of the examples, the data (and all the belonging parameters) could either be simulated
or read from an external file.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
from pandas.plotting import autocorrelation_plot


import trilearn.graph.decomposable as dlib
from trilearn import pgibbs
import trilearn.auxiliary_functions as aux
import trilearn.distributions.g_intra_class as gic
from trilearn.distributions import discrete_dec_log_linear as loglin


np.random.seed(2)

# Discrete data
# Load data from file
df = pd.read_csv("sample_data/czech_autoworkers.csv", header=[0, 1]) # read labels and number of levels from row 0 and 1
graph_traj = pgibbs.sample_trajectory_loglin(dataframe=df, n_particles=100, n_samples=100)

graph_traj.size().plot()
plt.savefig("sizetraj.png")
plt.clf()

graph_traj.log_likelihood().plot()
plt.savefig("logliktraj.png")
plt.clf()

aux.plot_heatmap(graph_traj.heatmap(), annot=True)
plt.savefig("heatmap.png")

top = graph_traj.empirical_distribution().mode(5)

for graph, prob in top:
    print graph.edges(), prob



## Continuous AR(1-5)-model
np.random.seed(2)
graph = dlib.sample_random_AR_graph(50, 5)
cov_mat = gic.cov_matrix(graph, 0.9, 1.0)
df = pd.DataFrame(np.random.multivariate_normal(np.zeros(50), cov_mat, 100))

graph_traj = pgibbs.sample_trajectory_ggm(dataframe=df, n_particles=50, n_samples=10,
                                          radius=5, alpha=0.5, beta=0.8,
                                          reset_cache=True)

graph_traj.size().plot()
plt.savefig("sizetraj.png")
plt.clf()

graph_traj.log_likelihood().plot()
plt.savefig("logliktraj.png")
plt.clf()

aux.plot_heatmap(graph_traj.empirical_distribution().heatmap())
plt.savefig("heatmap.png")
plt.clf()

graph_traj.write_file("pgibbs_AR1-5_p50_n100_N100_alpha_0_5_beta0_8_radius_5_M5000_traj.json")

top = graph_traj.empirical_distribution().mode(1)

aux.plot_heatmap(nx.to_numpy_array([0][0]))
plt.savefig("heatmap.png")
plt.clf()


## 15 nodes log-linear data
graph = nx.Graph()
graph.add_nodes_from(range(15))
graph.add_edges_from([(0, 11), (0, 7), (1, 8), (1, 6), (2, 4), (3, 8), (3, 9),
                       (3, 10), (3, 4), (3, 6), (4, 6), (4, 8), (4, 9), (4, 10),
                       (5, 10), (5, 6), (6, 8), (6, 9), (6, 10), (7, 11), (8, 9),
                       (8, 10), (8, 11), (9, 10), (10, 11), (12, 13)])

levels = np.array([range(2)] * graph.order())
table = loglin.sample_prob_table(graph, levels, 1.0)
assert(np.abs(table.sum() - 1.0) < 0.00001)

df = pd.DataFrame(loglin.sample(table, 100))
df.columns = [range(graph.order()), [len(l) for l in levels]]

graph_traj = pgibbs.sample_trajectory_loglin(dataframe=df, n_particles=20, alpha=0.2, beta=0.8, n_samples=1000)

graph_traj.size().plot()
plt.savefig("sizetraj.png")
plt.clf()

autocorrelation_plot(graph_traj.size())
plt.savefig("size_autocorr.png")
plt.clf()

graph_traj.log_likelihood().plot()
plt.savefig("logliktraj.png")
plt.clf()

aux.plot_heatmap(graph_traj.heatmap(), annot=True)
plt.savefig("heatmap.png")