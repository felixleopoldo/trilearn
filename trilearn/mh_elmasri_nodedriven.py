import numpy as np
import networkx as nx
from tqdm import tqdm

import trilearn.distributions.sequential_junction_tree_distributions as seqdist
import trilearn.graph.junction_tree as jtlib
import trilearn.graph.decomposable as dlib
import trilearn.graph.node_driven_moves as ndlib
import trilearn.graph.trajectory as mcmctraj


# starting MCMC sampler
def sample_trajectory(n_samples, randomize, sd, init_graph=None):
    if init_graph:
        graph = init_graph
    else:   
        graph = nx.Graph()
        graph.add_nodes_from(range(sd.p))
    # graph = ar_graph.copy()
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
            # print('MAP update....')
            MAP_graph = (graphs[i-1], log_prob_traj[i-1])
        if i % randomize == 0:
            jt = dlib.junction_tree(jtlib.graph(jt))
            jtlib.randomize(jt)
            graphs[i] = jtlib.graph(jt)  # TODO: Improve.
            jjt = dlib.junction_tree(jtlib.graph(jt))
            log_prob_traj[i] = sd.log_likelihood(graphs[i]) \
                - jtlib.log_n_junction_trees(jjt, jtlib.separators(jjt))
        # A move
        log_p1 = log_prob_traj[i - 1]
        node = np.random.randint(num_nodes)
        move_type = np.random.randint(2)
        if move_type == 0:          # connect
            new_cliques, log_q12, N, k = ndlib.propose_connect_moves(jt,
                                                                     node)
        else:                       # diconnect
            new_cliques, log_q12, N, k = ndlib.propose_disconnect_moves(jt,
                                                                        node)
        if new_cliques:
            jjt = dlib.junction_tree(jtlib.graph(jt))
            log_p2 = sd.log_likelihood(jtlib.graph(jt)) \
                - jtlib.log_n_junction_trees(jjt, jtlib.separators(jjt))
            # revese move
            log_q21, Ninv, kinv = ndlib.inverse_proposal_prob(jt,
                                                              node,
                                                              new_cliques,
                                                              move_type)
            #log_q12 = log_q21
            alpha = min(np.exp(log_p2 + log_q21 - log_p1 - log_q12), 1)
        else:
            alpha = 0
        # print alpha
        if np.random.uniform() <= alpha:
            # print "Accept"
            accept_traj[i] = 1
            log_prob_traj[i] = log_p2
            graphs[i] = jtlib.graph(jt)  # TODO: Improve.
        else:
            # print('reject')
            # print "Reject"
            # Reverse the tree
            ndlib.revert_moves(jt,
                               node,
                               new_cliques)
            log_prob_traj[i] = log_prob_traj[i-1]
            graphs[i] = graphs[i-1]

    gtraj.set_trajectory(graphs)
    gtraj.logl = log_prob_traj
    return gtraj


def sample_trajectory_ggm(dataframe, n_samples, randomize=1000,
                          D=None, delta=1.0, cache={}, **args):
    p = dataframe.shape[1]
    if D is None:
        D = np.identity(p)
    sd = seqdist.GGMJTPosterior()
    sd.init_model(np.asmatrix(dataframe), D, delta, cache)
    return sample_trajectory(n_samples, randomize, sd)


def max_likelihood_gmm(dataframe, graph, delta=1.0, jt=None):
    p = dataframe.shape[1]
    D = np.identity(p)
    sd = seqdist.GGMJTPosterior()
    sd.init_model(np.asmatrix(dataframe), D, delta, {})
    # graph = ar_graph.copy()
    if not jt:
        jt = dlib.junction_tree(graph)
        assert (jtlib.is_junction_tree(jt))
    else:
        graph = jtlib.graph(jt)

    loglike = sd.log_likelihood(graph) - \
        jtlib.log_n_junction_trees(jt, jtlib.separators(jt))

    return loglike
