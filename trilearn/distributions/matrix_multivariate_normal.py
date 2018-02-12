import numpy as np


def sample(M, S, Sigma):
    """ Generates a sample from the multivariate matrix
        normal distribution.
    """
    N = M.shape[0]
    K = M.shape[1]
    kron = np.kron(S, Sigma)
    vec_M = np.array(M.reshape(1, N*K))[0]
    vec_X = np.random.multivariate_normal(vec_M, kron)
    X = vec_X.reshape(M.shape)
    return X
