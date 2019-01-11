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

from trilearn.graph import trajectory
import trilearn.graph.decomposable as dlib
from trilearn import pgibbs
import trilearn.auxiliary_functions as aux
import trilearn.distributions.g_intra_class as gic
from trilearn.distributions import discrete_dec_log_linear as loglin


np.random.seed(2)

# Discrete data
# Load data from file
aw_df = pd.read_csv("sample_data/czech_autoworkers.csv", header=[0, 1]) # read labels and number of levels from row 0 and 1
aw_graph_traj = pgibbs.sample_trajectory_loglin(dataframe=df, n_particles=100, n_samples=10000)


top = aw_graph_traj.empirical_distribution().mode(5)

for graph, prob in top:
    print graph.edges(), prob

aw_graph_traj.size().plot()
plt.savefig(str(aw_graph_traj)+"_size.png")
plt.clf()

autocorrelation_plot(aw_graph_traj.size())
plt.savefig(str(aw_graph_traj)+"_autocorr.png")
plt.clf()

aw_graph_traj.log_likelihood().plot()
plt.savefig(str(aw_graph_traj)+"_loglik.png")
plt.clf()

aux.plot_heatmap(aw_graph_traj.empirical_distribution().heatmap())
plt.savefig(str(aw_graph_traj)+"_heatmap.png")
plt.clf()

top = aw_graph_traj.empirical_distribution().mode(1)
aux.plot_heatmap(nx.to_numpy_array(top[0][0]))
plt.savefig(str(aw_graph_traj)+"_map.png")
plt.clf()

aw_graph_traj.write_file(str(aw_graph_traj)+".json")


## Continuous AR(1-5)-model
np.random.seed(2)
ar_graph = dlib.sample_random_AR_graph(50, 5)
cov_mat = gic.cov_matrix(ar_graph, 0.9, 1.0)
ar_df = pd.DataFrame(np.random.multivariate_normal(np.zeros(50), cov_mat, 100))

ar_graph_traj = pgibbs.sample_trajectory_ggm(dataframe=ar_df, n_particles=50, n_samples=5000,
                                          radius=5, alpha=0.5, beta=0.5,
                                          reset_cache=True)

ar_graph_traj.size().plot()
plt.savefig(str(ar_graph_traj)+"_size.png")
plt.clf()

autocorrelation_plot(ar_graph_traj.size())
plt.savefig(str(ar_graph_traj)+"_autocorr.png")
plt.clf()

ar_graph_traj.log_likelihood().plot()
plt.savefig(str(ar_graph_traj)+"_loglik.png")
plt.clf()

aux.plot_heatmap(ar_graph_traj.empirical_distribution().heatmap())
plt.savefig(str(ar_graph_traj)+"_heatmap.png")
plt.clf()

ar_top = ar_graph_traj.empirical_distribution().mode(1)
aux.plot_heatmap(nx.to_numpy_array(top[0][0]))
plt.savefig(str(ar_graph_traj)+"_map.png")
plt.clf()

ar_graph_traj.write_file(str(ar_graph_traj)+".json")


# ## 15 nodes log-linear data
loglin_graph = nx.Graph()
loglin_graph.add_nodes_from(range(15))
loglin_graph.add_edges_from([(0, 11), (0, 7), (1, 8), (1, 6), (2, 4), (3, 8), (3, 9),
                       (3, 10), (3, 4), (3, 6), (4, 6), (4, 8), (4, 9), (4, 10),
                       (5, 10), (5, 6), (6, 8), (6, 9), (6, 10), (7, 11), (8, 9),
                       (8, 10), (8, 11), (9, 10), (10, 11), (12, 13)])

levels = np.array([range(2)] * loglin_graph.order())
loglin_table = loglin.sample_prob_table(loglin_graph, levels, 1.0)

loglin_df = pd.DataFrame(loglin.sample(loglin_table, 1000))
loglin_df.columns = [range(loglin_graph.order()), [len(l) for l in levels]]

loglin_graph_traj = pgibbs.sample_trajectory_loglin(dataframe=loglin_df, n_particles=20, alpha=0.2, beta=0.8, n_samples=10000)

loglin_graph_traj.size().plot()
plt.savefig(str(loglin_graph_traj)+"_size.png")
plt.clf()

autocorrelation_plot(loglin_graph_traj.size());
plt.savefig(str(loglin_graph_traj)+"_autocorr.png")
plt.clf()

loglin_graph_traj.log_likelihood().plot()
plt.savefig(str(loglin_graph_traj)+"_loglik.png")
plt.clf()

aux.plot_heatmap(loglin_graph_traj.empirical_distribution().heatmap())
plt.savefig(str(loglin_graph_traj)+"_heatmap.png")
plt.clf()

loglin_top = loglin_graph_traj.empirical_distribution().mode(1)
aux.plot_heatmap(nx.to_numpy_array(top[0][0]))
plt.savefig(str(loglin_graph_traj)+"_map.png")
plt.clf()

loglin_graph_traj.write_file(str(loglin_graph_traj)+".json")


# Multiple example
n_particles = [5]
n_samples = [10]
D=None
delta=1.0
alphas=[0.2, 0.5, 0.8]
betas=[0.2, 0.5, 0.8]
radii=[5, 50]
pgibbs.sample_trajectories_ggm_parallel(ar_df, n_particles, n_samples, D=D, delta=delta, alphas=alphas, betas=betas, radii=radii)

for N_i, N in enumerate(n_particles):
    for T_i, T in enumerate(n_samples):
        for radius_i, radius in enumerate(radii):
            for alpha_i, alpha in enumerate(alphas):
                for beta_i, beta in enumerate(betas):
                    traj = trajectory.Trajectory()
                    filename = ("pgibbs_graph_trajectory_ggm_posterior_" +
                    "n_"+str(ar_df.shape[0])+"_" +
                    "p_"+str(ar_df.shape[1])+"_" +
                    "prior_scale_"+str(delta)+"_"\
                    "shape_x_"\
                    "length_"+str(T)+"_"\
                    "N_"+str(N)+"_"\
                    "alpha_"+str(alpha)+"_"\
                    "beta_"+str(beta)+"_"\
                    "radius_"+str(radius)+".json")
                    traj.read_file(filename)
                    print str(traj)
                    traj.log_likelihood().plot();
                    plt.show();
                    plt.clf();
