"""
Sequential Monte Carlo sampler for junction tree distributions.
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import trilearn.distributions.sequential_junction_tree_distributions as seqdist
import trilearn.graph.graph as glib
import trilearn.graph.junction_tree as jtlib
import trilearn.graph.junction_tree_collapser
import trilearn.graph.junction_tree_expander
import trilearn.set_process as sp
import auxiliary_functions as aux

def smc_ggm_graphs(N, alpha, beta, radius, X, D, delta):
    cache = {}
    seq_dist = seqdist.GGMJTPosterior()
    seq_dist.init_model(X, D, delta, cache)
    (trees, log_w) = approximate(N, alpha, beta, radius, seq_dist)
    log_w_rescaled = np.array(log_w.T)[seq_dist.p - 1] - \
                     max(np.array(log_w.T)[seq_dist.p - 1])
    norm_w = np.exp(log_w_rescaled) / sum(np.exp(log_w_rescaled))
    graphs = [jtlib.graph(tree) for tree in trees]
    return (graphs, norm_w)


def smc_approximate_ggm(N, alpha, beta, radius, X, D, delta):
    (graphs, probs) = smc_ggm_graphs(N, alpha, beta, radius, X, D, delta)
    dist = {graphs[i]: probs[i] for i in range(len(graphs))}
    return dist


def approximate(N, alpha, beta, radius, seq_dist, debug=False, neig_set_cache={}):
    """ Sequential Monte Carlo for junction trees using the christmas
    tree algorithm as proposal kernel.

    Args:
        N (int): number
        alpha (float): sparsity parameter for the Christmas tree algorithm
        beta (float): sparsity parameter for the Christmas tree algorithm
        radius (float): defines the radius within which ned nodes are selected
        seqdist (SequentialJTDistributions): the distribution to be sampled from

    Returns:
        (new_trees, log_w)

    References:

    """
    p = seq_dist.p
    log_w = np.matrix(np.zeros((N, p)))
    new_trees = [None for _ in range(N)]
    old_trees = [None for _ in range(N)]
    ind_perms = np.matrix(np.zeros((N, p)), dtype=np.object)
    total = set(range(p))

    for n in range(p):
        norm_w = None
        new_trees = [None for _ in range(N)]
        if n > 0:
            log_w_rescaled = np.array(log_w.T)[n - 1] - max(np.array(log_w.T)[n - 1])
            norm_w = np.exp(log_w_rescaled) / sum(np.exp(log_w_rescaled))

        I = np.random.choice(N, size=N, p=norm_w)
        for i in range(N):
            if i % 5000 == 0 and not i == 0 and debug:
                print "n: " + str(n) + ", i: " + str(i)
            if n == 0:
                ind_perms[i, n] = sp.gen_order_neigh([], radius, total)
                node = ind_perms[i, n][n]
                T = jtlib.JunctionTree()
                T.add_node(frozenset([node]), label=tuple([node]), color="red")
                new_trees[i] = T
                log_w[i, n] = 0.0
            else:

                order_frozenset = frozenset(ind_perms[I[i], n - 1])
                if order_frozenset not in neig_set_cache:
                    neig_set_cache[order_frozenset] = sp.order_neigh_set(ind_perms[I[i], n - 1], radius, total)
                ind_perms[i, n] = ind_perms[I[i], n - 1] + [aux.random_element_from_coll(neig_set_cache[order_frozenset])]
                node = ind_perms[i, n][n]

                new_trees[i], K_st, old_cliques, old_separators, new_cliques, new_separators = trilearn.graph.junction_tree_expander.sample(
                    old_trees[I[i]], node, alpha, beta, only_tree=False)

                # Backward kernel
                log_R = trilearn.graph.junction_tree_collapser.log_pdf(new_trees[i], old_trees[I[i]], node)
                log_density_ratio = seq_dist.log_ratio(old_cliques,
                                                       old_separators,
                                                       new_cliques,
                                                       new_separators,
                                                       old_trees[I[i]], new_trees[i])
                log_w[i, n] = log_density_ratio + log_R - np.log(K_st)
        old_trees = new_trees
    return (new_trees, log_w)


def approximate_cond(N, alpha, beta, radius, seq_dist, T_cond, perm_cond, debug=False, neig_set_cache={}):
    """ SMC an junction trees conditioned on the trajectories T_cond
    and perm_cond.
    """
    p = seq_dist.p
    log_w = np.matrix(np.zeros((N, p)))
    Is = np.matrix(np.zeros((N, p)), dtype=int)

    old_trees = [None for _ in range(N)]
    new_trees = [None for _ in range(N)]
    ind_perms = np.matrix(np.zeros((N, p)), dtype=np.object)
    total = range(p)
    maxradius = radius >= p
    copy_time = 0.0

    for n in range(p):
        # Reset the new trees and perms so that we do not alter the old ones
        new_trees = [None for _ in range(N)]
        norm_w = None
        if n > 0:
            log_w_rescaled = np.array(log_w.T)[n - 1] - max(np.array(log_w.T)[n - 1])
            norm_w = np.exp(log_w_rescaled) / sum(np.exp(log_w_rescaled))

        I = np.random.choice(N, size=N, p=norm_w)
        for i in range(N):
            if i % 500 == 0 and not i == 0 and debug:
                print "n: " + str(n) + ", i: " + str(i)
            if n == 0:
                # Index permutation
                ind_perms[i, n] = sp.gen_order_neigh([], radius, total)
                node = ind_perms[i, n][n]
                T = jtlib.JunctionTree()
                T.add_node(frozenset(ind_perms[i, n]),
                           label=tuple([node]),
                           color="red")
                new_trees[i] = T
                log_w[i, n] = 0.0
            else:
                tmp = np.matrix(I).reshape((N, 1))
                Is[np.ix_(range(N), [n])] = tmp
                if i == 0:
                    # Weights for the fixed trajectory
                    T_old = T_cond[n - 1]
                    T = T_cond[n]
                    new_trees[i] = T
                    ind_perms[i, n] = perm_cond[n]

                    old_cliques = T_old.nodes()
                    old_separators = T_old.get_separators()
                    new_cliques = T.nodes()
                    new_separators = T.get_separators()
                    node = list(set(perm_cond[n]) - set(perm_cond[n - 1]))[0]
                    K_st = trilearn.graph.junction_tree_expander.pdf(T_old, T, alpha, beta, node)
                    log_order_pdf = sp.backward_order_neigh_log_prob(perm_cond[n - 1],
                                                                     perm_cond[n],
                                                                     radius, maxradius)
                    log_R = log_order_pdf + trilearn.graph.junction_tree_collapser.log_pdf(T, T_old, node)

                    # Set weight
                    log_w[i, n] = seq_dist.log_ratio(old_cliques,
                                                     old_separators,
                                                     new_cliques,
                                                     new_separators,
                                                     T_old,
                                                     T) + log_R - np.log(K_st)
                elif i > 0:
                    # Weights for rest
                    T_old = old_trees[I[i]]  # Create an nx.Graph once for speed.
                    # Get permutation
                    order_frozenset = frozenset(ind_perms[I[i], n - 1])
                    if order_frozenset not in neig_set_cache:
                        neig_set_cache[order_frozenset] = sp.order_neigh_set(ind_perms[I[i], n - 1], radius, total)

                    ind_perms[i, n] = ind_perms[I[i], n - 1] + [aux.random_element_from_coll(neig_set_cache[order_frozenset])]
                    node = ind_perms[i, n][n]  # the added node

                    # Expand the junction tree T
                    new_trees[
                        i], K_st, old_cliques, old_separators, new_cliques, new_separators = trilearn.graph.junction_tree_expander.sample(
                        T_old, node, alpha, beta, only_tree=False)
                    log_order_pr = sp.backward_order_neigh_log_prob(ind_perms[I[i], n - 1],
                                                                    ind_perms[i, n],
                                                                    radius, maxradius)
                    T = new_trees[i]
                    log_R = log_order_pr + trilearn.graph.junction_tree_collapser.log_pdf(T, T_old, node)
                    log_w[i, n] = seq_dist.log_ratio(old_cliques,
                                                     old_separators,
                                                     new_cliques,
                                                     new_separators,
                                                     T_old,
                                                     T) + log_R - np.log(K_st)
        old_trees = new_trees
    return (new_trees, log_w, Is)


def est_log_norm_consts(order, n_particles, sequential_distribution, alpha=0.5, beta=0.5, n_smc_estimates=1,
                        debug=False):
    log_consts = np.zeros(
        n_smc_estimates * (order)
    ).reshape(n_smc_estimates, (order))

    def estimate_norm_const(order, weights):
        log_consts = np.zeros(order)
        for n in range(1, order):
            log_consts[n] = log_consts[n - 1] + np.log(np.mean(weights[:, n]))

        return log_consts

    for t in tqdm(range(n_smc_estimates), desc="Const estimates"):
        (trees, log_w) = approximate(n_particles, alpha, beta, sequential_distribution.p, sequential_distribution)
        w = np.exp(log_w)
        log_consts[t, :] = estimate_norm_const(order, w)

        if debug:
            unique_trees = set()
            for tree in trees:
                tree_alt = (frozenset(tree.nodes()), frozenset([frozenset(e) for e in tree.edges()]))
                unique_trees.add(tree_alt)

            print("Sampled unique junction trees: " + str(len(unique_trees)))
            unique_graphs = set([glib.hash_graph(jtlib.graph(tree)) for tree in trees])

            print("Sampled unique chordal graphs: {n_unique_chordal_graphs}".format(
                n_unique_chordal_graphs=len(unique_graphs)),
            )

    if n_smc_estimates == 1:
        log_consts = log_consts.flatten()
    return log_consts


def est_n_dec_graphs(order, n_particles, alpha=0.5, beta=0.5, n_smc_estimates=1, debug=False):
    sd = seqdist.CondUniformJTDistribution(order)
    log_consts = est_log_norm_consts(order, n_particles, sd, alpha, beta, n_smc_estimates, debug)
    return np.exp(log_consts)


def uniform_dec_samples(order, n_particles, alpha=0.5, beta=0.5, debug=False):
    sd = seqdist.CondUniformJTDistribution(order)
    (trees, log_w) = approximate(n_particles, alpha, beta, sd.p, sd)
    graphs = [jtlib.graph(tree) for tree in trees]
    return graphs


def uniform_dec_maxl_clique_size_samples(order, n_particles, alpha=0.5, beta=0.5, debug=False):
    sd = seqdist.CondUniformJTDistribution(order)
    (trees, log_w) = approximate(n_particles, alpha, beta, sd.p, sd)
    w = np.exp(log_w[:, order - 1])
    norm_w = np.array(w / w.sum()).flatten()
    max_clique_sizes = []
    for tree in trees:
        clique_sizes = [len(clique) for clique in tree.nodes()]
        max_clique_sizes.append(max(clique_sizes))
    return np.array(max_clique_sizes), norm_w


def est_dec_max_clique_size(order, n_particles, alpha=0.5, beta=0.5, n_smc_estimates=1, debug=False):
    expected_maxl_clique_sizes = []
    for t in range(n_smc_estimates):
        if debug: print("Iteration: " + str(t + 1) + "/" + str(n_smc_estimates))
        max_clique_sizes, norm_w = uniform_dec_maxl_clique_size_samples(order, n_particles,
                                                                        alpha=alpha, beta=beta, debug=debug)
        est_exp = (max_clique_sizes * norm_w).sum()  # weighted expected value
        expected_maxl_clique_sizes.append(est_exp)
        if debug:
            print t, est_exp
    return expected_maxl_clique_sizes


def get_smc_trajs(Is):
    """ This method is made for visualizing the collapsing in SMC.
    """
    p = Is.shape[1]
    N = Is.shape[0]

    for i in reversed(range(N)):
        t = get_traj(p - 1, i, Is) + [i]
        if i == 0:
            plt.plot(range(p), t, color="r")
        else:
            plt.plot(range(p), t, color="b")
    plt.show()


def get_traj(n, i, Is):
    if n == 0:
        return []
    else:
        return get_traj(n - 1, Is[i, n], Is) + [Is[i, n]]
