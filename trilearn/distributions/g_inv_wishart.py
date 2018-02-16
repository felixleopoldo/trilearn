import numpy as np
from scipy.stats import invwishart

import trilearn.graph.decomposable
import trilearn.graph.graph as glib
from trilearn.distributions import matrix_multivariate_normal


def sample(G, dof, scale):
    """
    Sample from G-inverse Wishart distribution.

    C. M. Carvalho, H. Massam, and M. West. Simulation of hyper-inverse
    wishart distributions in graphical models. Biometrika, 94(3):647-659, 2007.

    Args:
        G (networkx graph): A decomposable graph
        scale (numpy matrix): Scale parameter, a positive definite matrix.
        delta (float): Degrees o freedom, a positive real number
    Returns:
        A sample from the G-inverse wishart distribution
    """
    (C, S, H, A, R) = trilearn.graph.decomposable.peo(G)
    p = len(G.nodes())
    sigma = np.matrix(np.zeros((p, p)))
    scale_c1 = scale[np.ix_(list(C[0]), list(C[0]))]
    sig_c1 = invwishart(dof, scale_c1).rvs(1)
    sigma[np.ix_(list(C[0]), list(C[0]))] = sig_c1

    for i in range(1, len(C)):
        scale_R = scale[np.ix_(list(R[i]), list(R[i]))]
        if len(S[i]) == 0:
            sigma[np.ix_(list(R[i]),
                         list(R[i]))] = invwishart(dof + len(R[i]),
                                                   scale_R).rvs(1)
            continue

        scale_RS = scale[np.ix_(list(R[i]), list(S[i]))]
        scale_SR = scale_RS.T
        scale_S = scale[np.ix_(list(S[i]), list(S[i]))]

        sig_R = sigma[np.ix_(list(R[i]), list(R[i]))]
        sig_RS = sigma[np.ix_(list(R[i]), list(S[i]))]
        sig_SR = sig_RS.T
        sig_S = sigma[np.ix_(list(S[i]), list(S[i]))]
        sig_SA = sigma[np.ix_(list(S[i]), list(A[i-1]))]

        # To be sampled: sig_R_cond_S = sig_R - sig_RS * np.inv(sig_S) * sig_SR
        scale_R_cond_S = scale_R - scale_RS * scale_S.I * scale_SR
        sig_R_cond_S = invwishart(dof + len(R[i]), scale_R_cond_S).rvs(1)
        U = matrix_multivariate_normal.sample(scale_RS * scale_S.I,
                                              sig_R_cond_S, scale_S.I)
        sig_RS = U * sig_S
        sig_R = sig_R_cond_S + sig_RS * sig_S.I * sig_SR

        # Set values
        sigma[np.ix_(list(R[i]), list(R[i]))] = sig_R
        sigma[np.ix_(list(R[i]), list(S[i]))] = sig_RS
        sigma[np.ix_(list(S[i]), list(R[i]))] = sig_RS.T

        sig_RA = sig_RS * sig_S.I * sig_SA
        sigma[np.ix_(list(R[i]), list(A[i-1]))] = sig_RA
        sigma[np.ix_(list(A[i-1]), list(R[i]))] = sig_RA.T
    return sigma
