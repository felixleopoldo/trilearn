import time
from multiprocessing.process import Process

import numpy as np
from tqdm import tqdm

import trilearn.graph
from trilearn import set_process as sp
from trilearn.distributions import sequential_junction_tree_distributions as seqdist
from trilearn.graph import trajectory as mcmctraj, junction_tree as jtlib
from trilearn.smc import approximate, approximate_cond


def sample_trajectory(smc_N, alpha, beta, radius, n_samples, seq_dist,
                      jt_traj=None, debug=False, reset_cache=True):
    """ A particle Gibbs implementation for approximating distributions over
    junction trees.

    Args:
        smc_N (int): Number of particles in SMC in each Gibbs iteration
        n_samples (int): Number of Gibbs iterations (samples)
        alpha (float): sparsity parameter for the Christmas tree algorithm
        beta (float): sparsity parameter for the Christmas tree algorithm
        radius (float): defines the radius within which ned nodes are selected
        seq_dist (SequentialJTDistributions): the distribution to be sampled from

    Returns:
        mcmctraj.Trajectory: Markov chain of teh underlying graphs of the junction trees sampled by pgibbs.
    """
    graph_traj = mcmctraj.Trajectory()
    graph_traj.set_sequential_distribution(seq_dist)
    neig_set_cache = {}
    (trees, log_w) = (None, None)
    prev_tree = None
    for i in tqdm(range(n_samples), desc="Particle Gibbs samples"):

        if reset_cache is True:
            seq_dist.cache = {}

        start_time = time.time()
        if i == 0:
            (trees, log_w) = approximate(smc_N, alpha, beta, radius, seq_dist, neig_set_cache=neig_set_cache)
        else:
            # Sample backwards trajectories
            perm_traj = sp.backward_perm_traj_sample(seq_dist.p, radius)
            T_traj = trilearn.graph.junction_tree_collapser.backward_jt_traj_sample(perm_traj,
                                                                                    prev_tree)
            (trees, log_w, Is) = approximate_cond(smc_N,
                                                  alpha,
                                                  beta,
                                                  radius,
                                                  seq_dist,
                                                  T_traj,
                                                  perm_traj, neig_set_cache=neig_set_cache)
        # Sample T from T_1..p
        log_w_rescaled = np.array(log_w.T)[seq_dist.p - 1] - max(np.array(log_w.T)[seq_dist.p - 1])
        norm_w = np.exp(log_w_rescaled) / sum(np.exp(log_w_rescaled))
        I = np.random.choice(smc_N, size=1, p=norm_w)[0]
        T = trees[I]
        end_time = time.time()
        prev_tree = T
        graph_traj.add_sample(jtlib.graph(T), end_time - start_time)

    return graph_traj


def sample_trajectory_ggm(dataframe, n_particles, n_samples, D=None, delta=1.0, alpha=0.5, beta=0.5,
                          radius=None, reset_cache=True, **args):

    """ Particle Gibbs for approximating distributions over
    Gaussian graphical models.

    Args:
        n_particles (int): Number of particles in SMC in each Gibbs iteration
        n_samples (int): Number of Gibbs iterations (samples)
        alpha (float): sparsity parameter for the Christmas tree algorithm
        beta (float): sparsity parameter for the Christmas tree algorithm
        radius (float): defines the radius within which ned nodes are selected
        dataframe (np.matrix): row matrix of data
        D (np.matrix): matrix parameter for the hyper inverse wishart prior
        delta (float): degrees of freedom for the hyper inverse wishart prior
        cache (dict): cache for clique likelihoods

    Returns:
        mcmctraj.Trajectory: Markov chain of the underlying graphs of the junction trees sampled by pgibbs.
    """

    p = dataframe.shape[1]
    if D is None:
        D = np.identity(p)
    if radius is None:
        radius = p
    sd = seqdist.GGMJTPosterior()
    sd.init_model(np.asmatrix(dataframe), D, delta, {})
    return sample_trajectory(n_particles, alpha, beta, radius, n_samples, sd, reset_cache=reset_cache)


def sample_trajectories_ggm(dataframe, n_particles, n_samples, D=None, delta=1.0, alphas=[0.5], betas=[0.5],
                            radii=[None], reset_cache=True, filename_prefix=None, **args):
    graph_trajectories = []
    for N in n_particles:
        for T in n_samples:
            for rad in radii:
                for alpha in alphas:
                    for beta in betas:
                        if rad is None:
                            rad = dataframe.shape[1]
                        graph_trajectory = sample_trajectory_ggm(dataframe, n_particles=N,
                                                                 n_samples=T, D=D, delta=delta,
                                                                 alpha=alpha, beta=beta,
                                                                 radius=rad, reset_cache=reset_cache)
                        graph_trajectories.append(graph_trajectory)
                        if filename_prefix:
                            graphs_file = filename_prefix + str(graph_trajectory) + '_N_' + str(N)
                            graphs_file += '_alpha_' + str(alpha) + '_beta_' + str(beta)
                            graphs_file += '_radius_' + str(rad) + '.txt'
                            graph_trajectory.write_file(graphs_file)
    return graph_trajectories


def sample_trajectories_ggm_parallel(dataframe, trajectory_lengths, n_particles,
                                     D=None, delta=1.0, alphas=[0.5], betas=[0.5], radii=[None],
                                     reset_cache=True,
                                     **args):
    p = dataframe.shape[1]
    if D is None:
        D = np.identity(p)
    if radii == [None]:
        radii = [p]
    for N in n_particles:
        for T in trajectory_lengths:
            for rad in radii:
                for alpha in alphas:
                    for beta in betas:
                        sd = seqdist.GGMJTPosterior()
                        sd.init_model(np.asmatrix(dataframe), D, delta, {})
                        print "Starting: " + str((N, alpha, beta, rad,
                                                  T, str(sd)))

                        proc = Process(target=trajectory_to_file,
                                       args=(N, alpha, beta, rad,
                                             T, sd))
                        proc.start()

    for N in n_particles:
        for T in trajectory_lengths:
            for rad in radii:
                for alpha in alphas:
                    for beta in betas:
                        proc.join()
                        print "Completed: " + str((N, alpha, beta,
                                                   rad, T))


def sample_trajectory_loglin(dataframe, n_particles, n_samples, pseudo_obs=1.0, reset_cache=False, alpha=0.5, beta=0.5,
                             radius=None, **args):
    p = dataframe.shape[1]
    if radius is None:
        radius = p

    n_levels = np.array(dataframe.columns.get_level_values(1), dtype=int)
    levels = np.array([range(l) for l in n_levels])

    sd = seqdist.LogLinearJTPosterior(dataframe.get_values(), pseudo_obs, levels, {})
    return sample_trajectory(n_particles, alpha, beta, radius, n_samples, sd, reset_cache=reset_cache)


def sample_trajectories_loglin(dataframe, trajectory_lengths, n_particles,
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
                            graph_trajectory = sample_trajectory_loglin(dataframe, N, T, pseudo_obs, cache, alpha,
                                                                        beta, rad)
                            graph_trajectories.append(graph_trajectory)
                            if filename_prefix:
                                if rad is None:
                                    rad = dataframe.shape[1]
                                graphs_file = filename_prefix + '_loglin_pseudo_obs_' + str(pseudo_obs) + '_T_' + str(
                                    T) + '_N_' + str(N)
                                graphs_file += '_alpha_' + str(alpha) + '_beta_' + str(beta)
                                graphs_file += '_radius_' + str(rad) + '_graphs.txt'
                                graph_trajectory.write_file(graphs_file)
    return graph_trajectories


def sample_trajectories_loglin_parallel(dataframe, trajectory_length, n_particles,
                                        pseudo_observations, alphas, betas, radii, filename_prefix,
                                        **args):
    cache = {}
    for N in n_particles:
        for T in trajectory_length:
            for rad in radii:
                for alpha in alphas:
                    for beta in betas:
                        for pseudo_obs in pseudo_observations:
                            sd = seqdist.LogLinearJTPosterior(dataframe, pseudo_obs, levels, cache)
                            print "Starting: " + str((N, alpha, beta, rad,
                                                      T, sd, filename_prefix,))
                            proc = Process(target=trajectory_to_file,
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


def trajectory_to_file(smc_N, alpha, beta, radius, n_samples,
                       seqdist, reset_cache=True):
    """ Writes the trajectory of graphs generated by particle Gibbs to file.

    Args:
        smc_N (int): Number of particles in SMC in each Gibbs iteration
        n_samples (int): Number of Gibbs iterations (samples)
        alpha (float): sparsity parameter for the Christmas tree algorithm
        beta (float): sparsity parameter for the Christmas tree algorithm
        radius (float): defines the radius within which ned nodes are selected
        seq_dist (SequentialJTDistributions): the distribution to be sampled from
        filename_prefix (string): prefix to the filename

    Returns:
        mcmctraj.Trajectory: Markov chain of underlying graphs of the junction trees sampled by pgibbs.

    """
    graphtraj = sample_trajectory(smc_N, alpha, beta, radius, n_samples, seqdist, reset_cache=reset_cache)

    graphs_file =  str(graphtraj) + '_N_' + str(smc_N)
    graphs_file += '_alpha_' + str(alpha) + '_beta_' + str(beta)
    graphs_file += '_radius_' + str(radius) + '.json'
    graphtraj.write_file(graphs_file)
    return graphtraj


# def particle_gibbs_ggm(X, alpha, beta, n_particles, traj_length, D, delta, radius, debug=False):
#     """ Particle Gibbs for approximating distributions over
#     Gaussian graphical models.
#
#     Args:
#         n_particles (int): Number of particles in SMC in each Gibbs iteration
#         traj_length (int): Number of Gibbs iterations (samples)
#         alpha (float): sparsity parameter for the Christmas tree algorithm
#         beta (float): sparsity parameter for the Christmas tree algorithm
#         radius (float): defines the radius within which ned nodes are selected
#         X (np.matrix): row matrix of data
#         D (np.matrix): matrix parameter for the hyper inverse wishart prior
#         delta (float): degrees of freedom for the hyper inverse wishart prior
#
#     Returns:
#         mcmctraj.Trajectory: Markov chain of the underlying graphs of the junction trees sampled by pgibbs.
#     """
#
#     cache = {}
#     seq_dist = seqdist.GGMJTPosterior()
#     seq_dist.init_model(np.asmatrix(X), D, delta, cache)
#     mcmctraj = particle_gibbs(n_particles, alpha, beta, radius, traj_length, seq_dist, debug=debug)
#     return mcmctraj
