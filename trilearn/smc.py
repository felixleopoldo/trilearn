"""
Sequential Monte Carlo sampler for junction tree distributions.
"""
import time
from multiprocessing import Process

import matplotlib.pyplot as plt
import numpy as np

import trilearn.distributions.sequential_junction_tree_distributions as seqdist
import trilearn.graph.christmas_tree_algorithm as jtexp
import trilearn.graph.graph as glib
import trilearn.graph.junction_tree as jtlib
import trilearn.graph.trajectory as mcmctraj
import trilearn.set_process as sp


def particle_gibbs(N, alpha, beta, radius, traj_length, seq_dist,
                   jt_traj=None):
    """ A particle Gibbs implementation for approximating distributions over
    junction trees.

    Args:
        N (int): Number of particles in SMC in each Gibbs iteration
        traj_length (int): Number of Gibbs iterations (samples)
        alpha (float): sparsity parameter for the Christmas tree algorithm
        beta (float): sparsity parameter for the Christmas tree algorithm
        radius (float): defines the radius within which ned nodes are selected
        seq_dist (SequentialJTDistributions): the distribution to be sampled from

    Returns:
        mcmctraj.Trajectory: Markov chain of teh underlying graphs of the junction trees sampled by pgibbs.
    """
    graph_traj = mcmctraj.Trajectory()
    graph_traj.set_sequential_distribution(seq_dist)

    (trees, log_w) = (None, None)
    prev_tree = None
    for i in range(traj_length):

        start_time = time.time()
        if i == 0:
            (trees, log_w) = smc(N, alpha, beta, radius, seq_dist)
        else:
            # Sample backwards trajectories
            perm_traj = sp.backward_perm_traj_sample(seq_dist.p, radius)
            T_traj = jtexp.backward_jt_traj_sample(perm_traj,
                                                   prev_tree)
            (trees, log_w, Is) = smc_cond(N,
            alpha,
            beta,
            radius,
            seq_dist,
            T_traj,
            perm_traj)

        # Sample T from T_1..p
        log_w_rescaled = np.array(log_w.T)[seq_dist.p-1] - max(np.array(log_w.T)[seq_dist.p-1])
        norm_w = np.exp(log_w_rescaled) / sum(np.exp(log_w_rescaled))
        I = np.random.choice(N, size=1, p=norm_w)[0]
        T = trees[I]
        end_time = time.time()
        prev_tree = T
        graph_traj.add_sample(jtlib.graph(T), end_time - start_time)
        print "PGibbs sample "+str(i+1)+"/"+str(traj_length) + \
            ". Sample time: " + str(end_time-start_time) + " s. " + \
            "Estimated time left: " + \
            str(int((end_time-start_time) * (traj_length - i - 1) / 60)) + " min."
    return graph_traj


def smc_graphs(N, alpha, beta, radius, X, D, delta):
    cache = {}
    seq_dist = seqdist.GGMJTPosterior()
    seq_dist.init_model(X, D, delta, cache)
    (trees, log_w) = smc(N, alpha, beta, radius, seq_dist)
    log_w_rescaled = np.array(log_w.T)[seq_dist.p-1] - \
        max(np.array(log_w.T)[seq_dist.p-1])
    norm_w = np.exp(log_w_rescaled) / sum(np.exp(log_w_rescaled))
    graphs = [jtlib.graph(tree) for tree in trees]
    return (graphs, norm_w)


def smc_approximate(N, alpha, beta, radius, X, D, delta):
    (graphs, probs) = smc_graphs(N, alpha, beta, radius, X, D, delta)
    dist = {graphs[i]: probs[i] for i in range(len(graphs))}
    return dist


def smc(N, alpha, beta, radius, seq_dist):
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
            log_w_rescaled = np.array(log_w.T)[n-1]-max(np.array(log_w.T)[n-1])
            norm_w = np.exp(log_w_rescaled) / sum(np.exp(log_w_rescaled))

        I = np.random.choice(N, size=N, p=norm_w)
        for i in range(N):
            if i % 5000 == 0 and not i == 0:
                print "n: "+str(n) + ", i: "+str(i)
            if n == 0:
                ind_perms[i, n] = sp.gen_order_neigh([], radius, total)
                node = ind_perms[i, n][n]
                T = jtlib.JunctionTree()
                T.add_node(frozenset([node]), label=tuple([node]), color="red")
                new_trees[i] = T
                log_w[i, n] = 0.0
            else:
                ind_perms[i, n] = sp.gen_order_neigh(ind_perms[I[i], n-1],
                                                     radius, total)
                node = ind_perms[i, n][n]
                new_trees[i], K_st, old_cliques, old_separators, new_cliques, new_separators = jtexp.expand(old_trees[I[i]], node, alpha, beta)
                # Backward kernel
                log_R = -jtexp.log_count_origins(new_trees[i], old_trees[I[i]], node)
                log_density_ratio = seq_dist.log_ratio(old_cliques,
                                                       old_separators,
                                                       new_cliques,
                                                       new_separators,
                                                       old_trees[I[i]], new_trees[i])
                log_w[i, n] = log_density_ratio + log_R - np.log(K_st)
        old_trees = new_trees
    return (new_trees, log_w)


def smc_cond(N, alpha, beta, radius, seq_dist, T_cond, perm_cond):
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
            log_w_rescaled = np.array(log_w.T)[n-1] - max(np.array(log_w.T)[n-1])
            norm_w = np.exp(log_w_rescaled) / sum(np.exp(log_w_rescaled))

        I = np.random.choice(N, size=N, p=norm_w)
        for i in range(N):
            if i % 500 == 0 and not i == 0:
                print "n: "+str(n) + ", i: "+str(i)
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
                    T_old = T_cond[n-1]
                    T = T_cond[n]
                    new_trees[i] = T
                    ind_perms[i, n] = perm_cond[n]

                    old_cliques = T_old.nodes()
                    old_separators = T_old.get_separators()
                    new_cliques = T.nodes()
                    new_separators = T.get_separators()
                    node = list(set(perm_cond[n]) - set(perm_cond[n-1]))[0]
                    K_st = jtexp.K_star(T_old, T, alpha, beta, node)
                    log_order_pr = sp.backward_order_neigh_log_prob(perm_cond[n-1],
                                                                    perm_cond[n],
                                                                    radius, maxradius)
                    log_R = log_order_pr - jtexp.log_count_origins(T, T_old, node)

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
                    ind_perms[i, n] = sp.gen_order_neigh(ind_perms[I[i], n-1],
                                                         radius,
                                                         total)
                    node = ind_perms[i, n][n]  # the added node
                    # Expand the junction tree T
                    new_trees[i], K_st, old_cliques, old_separators, new_cliques, new_separators = jtexp.expand(T_old, node, alpha, beta)
                    log_order_pr = sp.backward_order_neigh_log_prob(ind_perms[I[i], n-1],
                                                                    ind_perms[i, n],
                                                                    radius, maxradius)
                    T = new_trees[i]
                    log_R = log_order_pr - jtexp.log_count_origins(T, T_old, node)
                    log_w[i, n] = seq_dist.log_ratio(old_cliques,
                                                     old_separators,
                                                     new_cliques,
                                                     new_separators,
                                                     T_old,
                                                     T) + log_R - np.log(K_st)
        old_trees = new_trees
    return (new_trees, log_w, Is)


def est_norm_consts(order, n_particles, alpha=0.5, beta=0.5, n_smc_estimates=1, debug=False):
    sd = seqdist.CondUniformJTDistribution(order)

    log_consts = np.zeros(
        n_smc_estimates * order
    ).reshape(n_smc_estimates, order)

    def estimate_norm_const(order, weigths):
        log_consts = np.zeros(order)
        for n in range(1, order):
            log_consts[n] = log_consts[n - 1] + np.log(np.mean(weigths[:, n]))

        return log_consts

    for t in range(n_smc_estimates):
        if debug: print("Iteration: " + str(t + 1) + "/" + str(n_smc_estimates))
        (trees, log_w) = smc(n_particles, alpha, beta, sd.p, sd)
        w = np.exp(log_w)

        unique_trees = set()
        for tree in trees:
            tree_alt = (frozenset(tree.nodes()), frozenset([frozenset(e) for e in tree.edges()]))
            unique_trees.add(tree_alt)

        if debug: print("Sampled unique junction trees: " + str(len(unique_trees)))
        unique_graphs = set([glib.hash_graph(jtlib.graph(tree)) for tree in trees])

        if debug: print("Sampled unique chordal graphs: {n_unique_chordal_graphs}".format(
            n_unique_chordal_graphs=len(unique_graphs)),
        )

        for n in range(1, order):
            log_consts[t, n] = log_consts[t, n - 1] + np.log(np.mean(w[:, n]))

        if debug: print("Estimated number of chordal graphs: \n" + str(np.exp(log_consts[t, :])))
    return log_consts


def particle_gibbs_ggm(X, alpha, beta, n_particles, traj_length, D, delta, radius):
    """ Particle Gibbs for approximating distributions over
    Gaussian graphical models.

    Args:
        n_particles (int): Number of particles in SMC in each Gibbs iteration
        traj_length (int): Number of Gibbs iterations (samples)
        alpha (float): sparsity parameter for the Christmas tree algorithm
        beta (float): sparsity parameter for the Christmas tree algorithm
        radius (float): defines the radius within which ned nodes are selected
        X (np.matrix): row matrix of data
        D (np.matrix): matrix parameter for the hyper inverse wishart prior
        delta (float): degrees of freedom for the hyper inverse wishart prior

    Returns:
        mcmctraj.Trajectory: Markov chain of the underlying graphs of the junction trees sampled by pgibbs.
    """

    cache = {}
    seq_dist = seqdist.GGMJTPosterior()
    seq_dist.init_model(X, D, delta, cache)
    mcmctraj = particle_gibbs(n_particles, alpha, beta, radius, traj_length, seq_dist)
    return mcmctraj

def gen_pgibbs_ggm_trajectory(X, trajectory_length, n_particles,
                              D=None, delta=1.0, cache={}, alpha=0.5, beta=0.5, radius=None, **args):
    p = X.shape[1]
    if D is None:
        D = np.identity(p)
    if radius is None:
        radius = p
    sd = seqdist.GGMJTPosterior()
    sd.init_model(X, D, delta, cache)
    return particle_gibbs(n_particles, alpha, beta, radius, trajectory_length, sd)


def gen_pgibbs_loglin_trajectory(X, levels, trajectory_length, n_particles,
                                 pseudo_obs=1.0, cache={}, alpha=0.5, beta=0.5, radius=None, **args):
    p = X.shape[1]

    if radius is None:
        radius = p
    sd = seqdist.LogLinearJTPosterior(X, pseudo_obs, levels, cache)
    return particle_gibbs(n_particles, alpha, beta, radius, trajectory_length, sd)


def gen_pgibbs_ggm_trajectories(X, trajectory_lengths, n_particles,
                                D=None, delta=1.0, alphas=[0.5], betas=[0.5], radii=[None],
                                cache={}, filename_prefix=None,
                                **args):
    graph_trajectories = []
    for N in n_particles:
       for T in trajectory_lengths:
            for rad in radii:
                for alpha in alphas:
                    for beta in betas:
                        graph_trajectory = gen_pgibbs_ggm_trajectory(X, T, N,
                        D, delta, cache, alpha, beta, rad)
                        graph_trajectories.append(graph_trajectory)
                        if filename_prefix:
                            if rad is None:
                                rad = X.shape[1]
                            graphs_file = filename_prefix+'_ggm_jt_post_T_'+str(T)+'_N_'+str(N)
                            graphs_file += '_alpha_'+str(alpha)+'_beta_'+str(beta)
                            graphs_file += '_radius_'+str(rad)+'_graphs.txt'
                            graph_trajectory.write_file(graphs_file)
    return graph_trajectories


def gen_pgibbs_ggm_trajectories_parallel(X, trajectory_lengths, n_particles,
                                         D=None, delta=1.0, alphas=[0.5], betas=[0.5], radii=[None],
                                         cache={}, filename_prefix=None,
                                         **args):
    p = X.shape[1]
    if D is None:
        D = np.identity(p)
    if radii == [None]:
        radii = [p]
    cache = {}
    for N in n_particles:
        for T in trajectory_lengths:
            for rad in radii:
                for alpha in alphas:
                    for beta in betas:
                        sd = seqdist.GGMJTPosterior()
                        sd.init_model(X, D, delta, cache)
                        print "Starting: " + str((N, alpha, beta, rad,
                                                  T, sd, filename_prefix))

                        proc = Process(target=particle_gibbs_to_file,
                                       args=(N, alpha, beta, rad,
                                             T, sd, filename_prefix))
                        proc.start()

    for N in n_particles:
        for T in trajectory_lengths:
            for rad in radii:
                for alpha in alphas:
                    for beta in betas:
                        proc.join()
                        print "Completed: " + str((N, alpha, beta,
                                                   rad, T,
                                                   filename_prefix))


def gen_pgibbs_loglin_trajectories(X, levels, trajectory_lengths, n_particles,
                                   pseudo_observations=[1.0], alphas=[0.5], betas=[0.5], radii=[None],
                                   cache={}, filename_prefix=None,
                                   **args):
    graph_trajectories = []
    for N in n_particles:
       for T in trajectory_lengths:
            for rad in radii:
                for alpha in alphas:
                    for beta in betas:
                        for pseudo_obs in pseudo_observations:
                            graph_trajectory = gen_pgibbs_loglin_trajectory(X, levels, T, N, pseudo_obs,
                            cache, alpha, beta, rad)
                            graph_trajectories.append(graph_trajectory)
                            if filename_prefix:
                                if rad is None:
                                    rad = X.shape[1]
                                graphs_file = filename_prefix+'_loglin_pseudo_obs_'+str(pseudo_obs)+'_T_'+str(T)+'_N_'+str(N)
                                graphs_file += '_alpha_'+str(alpha)+'_beta_'+str(beta)
                                graphs_file += '_radius_'+str(rad)+'_graphs.txt'
                                graph_trajectory.write_file(graphs_file)
    return graph_trajectories


def gen_pgibbs_loglin_trajectories_parallel(X, levels, trajectory_length, n_particles,
    pseudo_observations, alphas, betas, radii, filename_prefix,
    **args):
    cache = {}
    for N in n_particles:
        for T in trajectory_length:
            for rad in radii:
                for alpha in alphas:
                    for beta in betas:
                        for pseudo_obs in pseudo_observations:
                            sd = seqdist.LogLinearJTPosterior(X, pseudo_obs, levels, cache)
                            print "Starting: " + str((N, alpha, beta, rad,
                            T, sd, filename_prefix,))
                            proc = Process(target=particle_gibbs_to_file,
                            args=(N, alpha, beta, rad,
                            T, sd, filename_prefix,))
                            proc.start()


    for N in n_particles:
        for T in trajectory_length:
            for rad in radii:
                for alpha in alphas:
                    for beta in betas:
                        for pseudo_obs in pseudo_observations:
                            proc.join()
                            print "Completed: " + str((N, alpha, beta,
                            rad, T,
                            filename_prefix))


def particle_gibbs_to_file(N, alpha, beta, radius, T,
                           seqdist, filename_prefix):
    """ Writes the trajectory of graphs generated by particle Gibbs to file.

    Args:
        N (int): Number of particles in SMC in each Gibbs iteration
        T (int): Number of Gibbs iterations (samples)
        alpha (float): sparsity parameter for the Christmas tree algorithm
        beta (float): sparsity parameter for the Christmas tree algorithm
        radius (float): defines the radius within which ned nodes are selected
        seq_dist (SequentialJTDistributions): the distribution to be sampled from
        filename_prefix (string): prefix to the filename

    Returns:
        mcmctraj.Trajectory: Markov chain of underlying graphs of the junction trees sampled by pgibbs.

    """
    graphtraj = particle_gibbs(N, alpha, beta, radius, T, seqdist)
    graphs_file = filename_prefix+'_'+str(seqdist) + '_T_'+str(T)+'_N_'+str(N)
    graphs_file += '_alpha_'+str(alpha)+'_beta_'+str(beta)
    graphs_file += '_radius_'+str(radius)+'_graphs.txt'
    graphtraj.write_file(graphs_file)
    return graphtraj


def get_smc_trajs(Is):
    """ This method is made for visualizing the collapsing in SMC.
    """
    p = Is.shape[1]
    N = Is.shape[0]
    for i in reversed(range(N)):
        t = get_traj(p-1, i, Is) + [i]
        if i == 0:
            plt.plot(range(p), t, color="r")
        else:
            plt.plot(range(p), t, color="b")
    plt.show()


def get_traj(n, i, Is):
    if n == 0:
        return []
    else:
        return get_traj(n-1, Is[i, n], Is) + [Is[i, n]]
