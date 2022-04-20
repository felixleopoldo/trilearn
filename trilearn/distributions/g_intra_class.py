"""
The graph intra-class distribution.
"""
import numpy as np

import trilearn.graph.decomposable
import trilearn.graph.graph as glib
import trilearn.graph.junction_tree as jtlib


def sample(G, r, s2, n):
    """ Samples from the G-intra-class distribution [1]_.

    Args:
        G (NetworkX graph): a decompoable graph
        r (float): correllation
        s2 (float): variance
        n (int): uber of samples

    Returns:
        np.matrix: n samples from the G-intra-class distribution in a row matrix.

    References:
        .. [1] P. J. Green and A. Thomas. Sampling decomposable graphs using a Markov chain on junction trees. Biometrika, 2013. https://doi.org/10.1093/biomet/ass052

    """
    (C, S, H, A, R) = trilearn.graph.decomposable.peo(G)
    p = G.order()
    I = np.matrix(np.identity(p))
    J = np.matrix(np.ones((p, p)))
    y = np.zeros(p)
    X = np.matrix(np.zeros((n, p)))

    IC = I[np.ix_(list(C[0]), list(C[0]))]
    JC = J[np.ix_(list(C[0]), list(C[0]))]

    for j in range(n):
        var = s2 * (1-r) * IC + r * JC
        y = np.zeros(p)
        y[list(C[0])] = np.random.multivariate_normal(np.zeros(len(C[0])), var)
        for i in range(1, len(S)):
            vs = len(S[i])
            IR = I[np.ix_(list(R[i]), list(R[i]))]
            JR = J[np.ix_(list(R[i]), list(R[i]))]
            M = r / (1.0 - r + vs*r)
            M *= sum(y[list(S[i])]) * np.ones((len(R[i]), 1))
            var = (1.0-r) * s2 * (IR + (r / (1.0 - r + vs * r)) * JR)
            y[list(R[i])] = np.random.multivariate_normal(np.array(M)[0], var)
        X[np.ix_([j])] = y
    return X.T


def cov_matrix(G, r, s2):
    """ Returns a covariance matrix cov such that zeros in cov.I is determined by G.

    Args:
        G (NetworkX graph): A decomposable graph.
        r (float): Correlation.
        s2 (float): Variance.

    Returns:
        Numpy matrix: A covariance matrix cov such that zeros in it inverse is determined by G.
    """
    p = G.order()
    T = trilearn.graph.decomposable.junction_tree(G)
    cliques = T.nodes()
    seps = jtlib.separators(T)
    omega = np.matrix(np.zeros((p, p)))
    cov = np.matrix(np.zeros((p, p)))
    for c in cliques:
        l = len(c)
        cov[np.ix_(list(c), list(c))] += np.identity(l) * s2
        cov[np.ix_(list(c),
                   list(c))] += (np.zeros((l, l)) + 1 - np.identity(l))*s2*r

    for s in seps:
        l = len(s)
        if l == 0:
            continue

        ls = len(seps[s])
        cov[np.ix_(list(s), list(s))] -= ls * np.identity(l) * s2
        cov[np.ix_(list(s),
                   list(s))] -= ls * (np.zeros((l, l)) + 1 - np.identity(l)) * s2 * r

    for c in cliques:
        l = len(c)
        omega[np.ix_(list(c), list(c))] += cov[np.ix_(list(c), list(c))].I

    for s in seps:
        l = len(s)
        if l == 0:
            continue
        ls = len(seps[s])
        omega[np.ix_(list(s),
                     list(s))] -= ls * cov[np.ix_(list(s), list(s))].I

    return omega.I
