"""
Metropolis-Hastings sampler for junction tree distributions.
"""
import time
import random
from multiprocessing import Process
from decimal import Decimal

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import trilearn.distributions.sequential_junction_tree_distributions as seqdist
import trilearn.distributions.gaussian_graphical_model as ggm
import trilearn.graph.graph as glib
import trilearn.graph.junction_tree as jtlib
import trilearn.graph.junction_tree_collapser
import trilearn.graph.junction_tree_expander
import trilearn.graph.trajectory as mcmctraj
import trilearn.set_process as sp


def gen_ggm_trajectory(dataframe, n_samples, D=None, delta=1.0, cache={}, alpha=0.5, beta=0.5, **args):
    p = dataframe.shape[1]
    if D is None:
        D = np.identity(p)
    sd = seqdist.GGMJTPosterior()
    sd.init_model(np.asmatrix(dataframe), D, delta, cache)
    return mh(alpha, beta, n_samples, sd)


def mh(alpha, beta, traj_length, seq_dist,
    jt_traj=None, debug=False):
    """ A Metropolis-Hastings implementation for approximating distributions over
    junction trees.

    Args:
        traj_length (int): Number of Gibbs iterations (samples)
        alpha (float): sparsity parameter for the Christmas tree algorithm
        beta (float): sparsity parameter for the Christmas tree algorithm
        seq_dist (SequentialJTDistributions): the distribution to be sampled from

    Returns:
        mcmctraj.Trajectory: Markov chain of teh underlying graphs of the junction trees sampled by M-H.
    """
    graph_traj = mcmctraj.Trajectory()
    graph_traj.set_sequential_distribution(seq_dist)

    prev_tree = None
    for i in tqdm(range(traj_length), desc="Metropolis-Hastings samples"):
        tree = None
        start_time = time.time()
        if i == 0:
            tree = jtlib.sample(seq_dist.p, alpha, beta)
        else:
            # Sample backwards trajectories
            tree = trans_sample(prev_tree, alpha, beta, seq_dist)
        # Sample T from T_1..p
        end_time = time.time()
        graph_traj.add_sample(jtlib.graph(tree), end_time - start_time)
        prev_tree = tree

    return graph_traj


def trans_sample(from_tree, alpha, beta, seq_dist, **args):

    prop_tree, reduced_tree, moved_node = proposal_sample(from_tree, alpha, beta, seq_dist.p)
    acc_prob = accept_proposal_prob(from_tree, reduced_tree, prop_tree, moved_node, alpha, beta, seq_dist)

    if np.random.binomial(1, acc_prob):
        return prop_tree
    else:
        return from_tree


def proposal_sample(from_tree, alpha, beta, n_nodes, **args):

    node = np.random.randint(n_nodes)
    reduced_tree = trilearn.graph.junction_tree_collapser.sample(from_tree, node)
    new_tree, K_st, old_cliques, old_separators, new_cliques, new_separators = trilearn.graph.junction_tree_expander.sample(
        reduced_tree, node, alpha, beta, only_tree=False)

    return new_tree, reduced_tree, node


def log_prop_pdf(from_tree, reduced_tree, to_tree, moved_node, alpha, beta):
    # Sum over R(to_tree, tree)K(tree, from_tree) for tree in Supp(R(to_tree, .)) = Supp(K(., to_tree))

    #log_prob = 0
    #for origin in trilearn.graph.junction_tree_collapser.possible_origins(from_tree, moved_node):
    #    log_prob += -np.log(sum(origin))
    log_prob = trilearn.graph.junction_tree_collapser.log_pdf(from_tree, reduced_tree, node=moved_node)
    log_prob += np.log(trilearn.graph.junction_tree_expander.pdf(reduced_tree, to_tree, alpha, beta, moved_node))
    return log_prob


def log_prop_ratio(from_tree, reduced_tree, to_tree, moved_node, alpha, beta):
    # print "from tree nodes: " + str(from_tree.nodes())
    # print "proposed tree nodes: " + str(to_tree.nodes())
    # print "moved node: " + str(moved_node)
    # print "reduced tree nodes: " + str(reduced_tree.nodes())

    ret = log_prop_pdf(to_tree, reduced_tree, from_tree, moved_node, alpha, beta)
    ret -= log_prop_pdf(from_tree, reduced_tree, to_tree, moved_node, alpha, beta)
    return ret


def log_post_ratio(from_tree, to_tree, seqdist):
    from_tree_seps = jtlib.separators(from_tree)
    log_post1 = seqdist.ll_partial(from_tree.nodes(), from_tree_seps)
    log_post1 -= jtlib.log_n_junction_trees(from_tree, from_tree_seps)

    to_tree_seps = jtlib.separators(to_tree)
    log_post2 = seqdist.ll_partial(to_tree.nodes(), to_tree_seps)
    log_post2 -= jtlib.log_n_junction_trees(to_tree, to_tree_seps)

    return log_post2 - log_post1


def accept_proposal_prob(from_tree, reduced_tree, to_tree, moved_node, alpha, beta, seq_dist):
    log_acc_prob = log_prop_ratio(from_tree, reduced_tree, to_tree, moved_node, alpha, beta)
    log_acc_prob += log_post_ratio(from_tree, to_tree, seq_dist)

    return min(1.0, np.exp(log_acc_prob))