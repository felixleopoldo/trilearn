"""
Examples for graph inference for continuous data and discrete data.
The data (and all the belonging parameters) are either be simulated
or read from an external file.
"""

import numpy as np
import pandas as pd
import networkx as nx

from trilearn import pgibbs
import trilearn.auxiliary_functions as aux
import trilearn.graph.decomposable as dlib
import trilearn.distributions.g_intra_class as gic
import trilearn.mh_greenthomas as green
import trilearn.distributions.discrete_dec_log_linear as loglin

np.random.seed(2)

# Discrete data
# reads labels and support from rows 0 and 1 respectively
aw_df = pd.read_csv("sample_data/czech_autoworkers.csv", header=[0, 1])
pgibbs.sample_trajectories_loglin_parallel(dataframe=aw_df, n_particles=[100], n_samples=[10000], reps=10,
                                           output_directory="./aw_trajs/")
graph_trajs = aux.read_all_trajectories_in_dir("./aw_trajs/")
aux.plot_multiple_traj_statistics(graph_trajs, 0, write_to_file=True, output_directory="./aw_trajs/")

## Continuous AR(1-5)-model
np.random.seed(2)
ar_graph = dlib.sample_random_AR_graph(50, 5)
cov_mat = gic.cov_matrix(ar_graph, 0.9, 1.0)
ar_df = pd.DataFrame(np.random.multivariate_normal(np.zeros(50), cov_mat, 100))

# PGibbs algorithm
pgibbs.sample_trajectories_ggm_parallel(dataframe=ar_df, n_particles=[50], n_samples=[10000],
                                        radius=[5, 50], alpha=[0.5], beta=[0.8],
                                        reset_cache=True, reps=10,
                                        output_directory="./ar_1-5_trajs/")
graph_trajs = aux.read_all_trajectories_in_dir("./ar_1-5_trajs/")
aux.plot_multiple_traj_statistics(graph_trajs, 0, write_to_file=True, output_directory="./ar_1-5_trajs/")

# Metropolis-Hastings algorithm from
# P. J. Green and A. Thomas. Sampling decomposable graphs using a Markov chain on junction trees. Biometrika, 100(1):91-110, 2013.
green.sample_trajectories_ggm_parallel(dataframe=ar_df, randomize=[100, 1000], n_samples=[500000],
                                        reset_cache=True, reps=10, output_directory="./ar_1-5_trajs_mh/")
graph_trajs = aux.read_all_trajectories_in_dir("./ar_1-5_trajs_mh/")
aux.plot_multiple_traj_statistics(graph_trajs, 0, write_to_file=True, output_directory="./ar_1-5_trajs_mh/")


## 15 nodes log-linear data
loglin_graph = nx.Graph()
loglin_graph.add_nodes_from(range(15))
loglin_graph.add_edges_from([(0, 11), (0, 7), (1, 8), (1, 6), (2, 4), (3, 8), (3, 9),
                       (3, 10), (3, 4), (3, 6), (4, 6), (4, 8), (4, 9), (4, 10),
                       (5, 10), (5, 6), (6, 8), (6, 9), (6, 10), (7, 11), (8, 9),
                       (8, 10), (8, 11), (9, 10), (10, 11), (12, 13)])
np.random.seed(1)
levels = np.array([range(2)] * loglin_graph.order())
loglin_table = loglin.sample_prob_table(loglin_graph, levels, 1.0)
np.random.seed(5)
loglin_df = pd.DataFrame(loglin.sample(loglin_table, 1000))
loglin_df.columns = [range(loglin_graph.order()), [len(l) for l in levels]]

pgibbs.sample_trajectories_loglin_parallel(dataframe=loglin_df, n_particles=[100], n_samples=[10000], reps=10,
                                           output_directory="./loglin_trajs/")
graph_trajs = aux.read_all_trajectories_in_dir("./loglin_trajs/")
aux.plot_multiple_traj_statistics(graph_trajs, 0, write_to_file=True, output_directory="./loglin_trajs/")
