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
from networkx.readwrite import json_graph


import trilearn.distributions.g_inv_wishart as gwish
import trilearn.graph.decomposable as dlib
import trilearn.graph.graph
import trilearn.smc as smc
from trilearn import pgibbs
from trilearn import mh_nodedriven
from  trilearn import mh_greenthomas
import trilearn.auxiliary_functions as aux
import trilearn.distributions.g_intra_class as gic
from trilearn.distributions import discrete_dec_log_linear as loglin


np.random.seed(2)

# ## Discrete data
# Load data from file
df = pd.read_csv("sample_data/czech_autoworkers.csv", header=[0, 1]) # read labels and number of levels from row 0 and 1
graph_traj = pgibbs.sample_trajectory_loglin(dataframe=df, n_particles=50, n_samples=1000)

graph_traj.size().plot()
plt.savefig("sizetraj.png")
plt.clf()

graph_traj.log_likelihood().plot()
plt.savefig("logliktraj.png")
plt.clf()

aux.plot_heatmap(graph_traj.heatmap())
plt.savefig("heatmap.png")


## Continuous AR(1-5)-model

graph = dlib.sample_random_AR_graph(50, 5)
cov_mat = gic.cov_matrix(graph, 0.9, 1.0)
df = pd.DataFrame(np.random.multivariate_normal(np.zeros(50), cov_mat, 100))

graph_traj = pgibbs.sample_trajectory_ggm(dataframe=df, n_particles=50, n_samples=100, debug=True)
graph_traj.size().plot()
plt.savefig("sizetraj.png")
plt.clf()

graph_traj.log_likelihood().plot()
plt.savefig("logliktraj.png")
plt.clf()

aux.plot_heatmap(graph_traj.heatmap())
plt.savefig("heatmap.png")


## 15 nodes log-linear data

## Load graph
graph = nx.Graph()
graph.add_nodes_from(range(15))
graph.add_edges_from([(0, 11), (0, 7), (1, 8), (1, 6), (2, 4), (3, 8), (3, 9),
                      (3, 10), (3, 4), (3, 6), (4, 6), (4, 8), (4, 9), (4, 10),
                      (5, 10), (5, 6), (6, 8), (6, 9), (6, 10), (7, 11), (8, 9),
                      (8, 10), (8, 11), (9, 10), (10, 11), (12, 13)])

levels = np.array([range(2)] * graph.order())
local_tables = loglin.sample_hyper_consistent_parameters(graph, 1.0,
                                                         levels)
print "local tables"
print local_tables
table = loglin.joint_prob_table(graph, local_tables, levels)
print "table"
print table
df = pd.DataFrame(loglin.sample(table, 100))

print df
## Load data from file

df = pd.read_csv("sample_data/dataset_p15.csv", header=None)


graph_traj = pgibbs.sample_trajectory_ggm(dataframe=df, n_particles=50, n_samples=10)
graph_traj.size().plot()
plt.show()
graph_traj.log_likelihood().plot()
plt.show()
graph_traj.heatmap()
plt.show()

## Continuous data
# Load data from file
df = pd.read_csv("sample_data/thomasgreen_p50_n400.csv", sep=' ', header=None)
graph_traj = mh_greenthomas.gen_ggm_trajectory(df, 3000)
graph_traj.size().plot()
plt.title("Graph size")
plt.savefig("mh_thomasgreen_p50_n400_size.png")
# plt.show()
plt.clf()
graph_traj.log_likelihood().plot()
plt.title("Log-likelihood")
plt.savefig("thomasgreen_p50_n400_mh_likelihood.png")
#plt.show()

plt.clf()
aux.heatmap(graph_traj.heatmap())
plt.title("Heatmap")
plt.savefig("thomasgreen_p50_n400_mh_heatmap.png")
#plt.show()


## Continuous data

## Load graph

## Generate precision matrix

## Simulate data

# Load data from file
df = pd.read_csv("sample_data/thomasgreen_p50_n400.csv", sep=' ', header=None)
graph_traj = mh_nodedriven.gen_ggm_trajectory(df, 30000)
graph_traj.size().plot()
plt.title("Graph size")
plt.savefig("mh_thomasgreen_data_p50_n400_size.png")
# plt.show()
plt.clf()
graph_traj.log_likelihood().plot()
plt.title("Log-likelihood")
plt.savefig("mh_thomasgreen_data_p50_n400_mh_likelihood.png")
#plt.show()

plt.clf()
aux.heatmap(graph_traj.heatmap())
plt.title("Heatmap")
plt.savefig("thomasgreen_data_p50_n400_mh_heatmap.png")
#plt.show()


## Continuous data

## Load graph
with open("sample_data/graph_p15.json") as data_file:
    js_graph = json.load(data_file)

graph = json_graph.node_link_graph(js_graph)

## Generate intraclass matrix
cov_mat = gic.cov_matrix(graph, 0.5, 1.0)

## Generate data
df = pd.DataFrame(np.random.multivariate_normal(np.zeros(graph.order()), cov_mat, 100))

## Create

# Load data from file
df = pd.read_csv("sample_data/dataset_p15.csv", header=None)
graph_traj = mh.gen_ggm_trajectory(df, 30000)
graph_traj.size().plot()
plt.title("Graph size")
plt.savefig("mh_size.png")
# plt.show()
plt.clf()
graph_traj.log_likelihood().plot()
plt.title("Log-likelihood")
plt.savefig("mh_likelihood.png")
#plt.show()

plt.clf()
aux.heatmap(graph_traj.heatmap())
plt.title("Heatmap")
plt.savefig("mh_heatmap.png")
#plt.show()

#
# # Sample synthetic data
# p = 5
# n_datasamples = 1000
# # sample graph
# graph = trilearn.graph.decomposable.sample_decomposable_graph(p)
# #glib.plot_adjmat(graph)
# #plt.show()
# # sample precision matrix
# omega = gwish.sample(graph, p, np.matrix(np.identity(p)))
# # sample normal data
# x = np.random.multivariate_normal(np.zeros(p), omega.I, n_datasamples)
# # generate pgibbs trajectory
# df = pd.DataFrame(x)
# graph_traj = smc.gen_pgibbs_ggm_trajectory(dataframe=df, n_particles=50, n_samples=10, debug=True)
# graph_traj.autocorrelation_size().plot()
# plt.show()
# graph_traj.likelihood().plot()
# plt.show()
# graph_traj.plot_heatmap()
# plt.show()
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