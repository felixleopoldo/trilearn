"""
Gaussian graphical model.
"""

import numpy as np

import trilearn.graph.decomposable
from trilearn.distributions import wishart as wish
import trilearn.graph.graph as glib
import trilearn.graph.junction_tree as jtlib


def log_likelihood(graph, S, n, D, delta, cache={}):
    """

    Args:
        S (Numpy matrix): sum of squares matrix for the full distribution
        D (Numpy matrix): location matrix for the full distribution
        delta (float): scale parameter
        n (int): number of data samples on which S is built
    """
    tree = trilearn.graph.decomposable.junction_tree(graph)
    separators = jtlib.separators(tree)
    cliques = tree.nodes()
    return log_likelihood_partial(S, n, D, delta, cliques, separators, cache)


def gaussian_marginal_log_likelihood(S, n, D, delta):
    """ Marginal log-likelihood of the data, x in a normal distribution
    with zero mean and where the precision matrix is marginalized out.

    Args:
        S (Numpy matrix): sum of squares matrix for the full distribution
        D (Numpy matrix): location matrix for the full distribution
        delta (float): scale parameter
        n (int): number of data samples on which S is built
    """
    c1 = wish.log_norm_constant(D+S, delta+n)
    c2 = wish.log_norm_constant(D, delta)
    return c1 - c2


def log_likelihood_partial(S, n, D, delta, cliques, separators, cache={}, idmatrices=None):
    """ Partial log-likelihood of the given cliques and separators.
    If every clique an separator is found in a graph, this is
    the marginal likelihood of that graph.

    Args:
        S (Numpy matrix): sum of squares matrix for the full distribution
        D (Numpy matrix): location matrix for the full distribution
        delta (float): scale parameter
        n (int): number of data samples on which S is built
        cliques (list): list of cliques, represented as frozensets
        separators (dict): dict with separaters as keys and list of associated edges as values
        cache (dict): dict of seps of cliques as kayes and partial ll as values
    """
    #if idmatrices is []:
    #    idmatrices = [np.identity(i) for i in range(D.shape[0]+1)]

    cliques_constants = 0.0
    for c in cliques:
        if c not in cache:
            inds = np.array(list(c), dtype=int).tolist()
            D_c = D[inds][:, inds]
            S_c = S[inds][:, inds]
            cache[c] = gaussian_marginal_log_likelihood(S_c, n, D_c, delta)
        cliques_constants += cache[c]
    seps_constants = 0.0
    for s in separators:
        if s == frozenset({}):
            continue
        if s not in cache:
            inds = np.array(list(s), dtype=int).tolist()
            D_s = D[inds][:, inds]
            S_s = S[inds][:, inds]
            cache[s] = gaussian_marginal_log_likelihood(S_s, n, D_s, delta)
        nu = len(separators[s])
        seps_constants += nu * cache[s]

    return cliques_constants - seps_constants
