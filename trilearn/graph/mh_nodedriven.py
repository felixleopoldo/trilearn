from multiprocessing import Process
import multiprocessing
import datetime
import time
import os

import numpy as np
import networkx as nx
from tqdm import tqdm

import trilearn.distributions.sequential_junction_tree_distributions as seqdist
import trilearn.graph.junction_tree as jtlib
import trilearn.graph.decomposable as dlib
import trilearn.graph.node_driven_moves as ndlib


from importlib import reload 
import matplotlib.pyplot as plt
import pandas as pd
import trilearn.graph.trajectory as mcmctraj
from trilearn import auxiliary_functions as aux

jtlib = reload(jtlib)
ndlib = reload(ndlib)
dlib = reload(dlib)


# starting MCMC sampler
np.random.seed(2)
dataframe = pd.read_csv("sample_data/data_ar1-5.csv", header=[0, 1])
dataframe = pd.read_csv('sample_data/greenthomas2009graph.csv', header=[0,1])
n_samples = 10000
randomize = 100
delta = 1
p = dataframe.shape[1]
D = np.identity(p)
sd = seqdist.GGMJTPosterior()
cache = {}
sd.init_model(np.asmatrix(dataframe), D, delta, cache)


graph = nx.Graph()
graph.add_nodes_from(range(sd.p))
jt = dlib.junction_tree(graph)
assert (jtlib.is_junction_tree(jt))
jt_traj = [None] * n_samples
graphs = [None] * n_samples
jt_traj[0] = jt
graphs[0] = jtlib.graph(jt)
log_prob_traj = [None] * n_samples

gtraj = mcmctraj.Trajectory()
gtraj.set_sampling_method({"method": "mh",
                           "params": {"samples": n_samples,
                                      "randomize_interval": randomize}
                           })

gtraj.set_sequential_distribution(sd)

log_prob_traj[0] = 0.0
log_prob_traj[0] = sd.log_likelihood(jtlib.graph(jt_traj[0]))
log_prob_traj[0] += -jtlib.log_n_junction_trees(jt_traj[0],
                                                jtlib.separators(jt_traj[0]))
accept_traj = [0] * n_samples

MAP_graph = (graphs[0], log_prob_traj[0])
num_nodes = len(MAP_graph[0])

for i in tqdm(range(1, n_samples), desc="Metropolis-Hastings samples"):
    if log_prob_traj[i-1] > MAP_graph[1]:
        MAP_graph = (graphs[i-1], log_prob_traj[i-1])

    if i % randomize == 0:
        jtlib.randomize(jt)
        graphs[i] = jtlib.graph(jt)  # TODO: Improve.
        log_prob_traj[i] = sd.log_likelihood(graphs[i]) \
            - jtlib.log_n_junction_trees(jt, jtlib.separators(jt))
    # A move
    log_p1 = log_prob_traj[i - 1]
    sample_node = np.random.randint(num_nodes) + 1
    new_nodes, log_q12 = ndlib.propose_moves(jt, sample_node)
    log_p2 = sd.log_likelihood(jtlib.graph(jt)) - jtlib.log_n_junction_trees(
        jt, jtlib.separators(jt))
    # revese move
    log_q21, revn, revk, revm = ndlib.inverse_proposal_prob(new_nodes,
                                                            jt, sample_node)
    alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
    # print alpha
    if np.random.uniform() <= alpha:
        # print "Accept"
        print('accept')
        accept_traj[i] = 1
        log_prob_traj[i] = log_p2
        graphs[i] = jtlib.graph(jt)  # TODO: Improve.
    else:
        print('reject')
        # print "Reject"
        # Reverse the tree
        ndlib.revert_moves(new_nodes, jt, sample_node)
        log_prob_traj[i] = log_prob_traj[i-1]
        graphs[i] = graphs[i-1]


gtraj.set_trajectory(graphs)
gtraj.logl = log_prob_traj

