import numpy as np
import scipy.special as scp


def normalizing_constant(phi, delta):
    return np.exp(log_norm_constant(phi, delta))


def logpdf(S, D, delta):
    p = S.shape[0]
    const = log_norm_constant(D, delta)
    (sign, logdet) = np.linalg.slogdet(S)
    tmp = -logdet * (delta + 2 * p) * 0.5 - (S.I * D).trace().item(0) * 0.5
    return tmp - const


def log_norm_constant(D, delta, cache={}):

    # This parametrization makes the requirement
    # delta > 0 instead of delta > p-1
    K = 0.0
    p = len(D)
    t = (delta + p - 1.0) / 2.0
    K = np.log(2) * (delta * p / 2.0)
    if (t, p) not in cache:
        cache[(t, p)] = scp.multigammaln(t, p)
    K += cache[(t, p)]

    tup = tuple(np.array(D).ravel())
    if tup not in cache:
        (sign, logdet) = np.linalg.slogdet(D)
        cache[tup] = logdet
    K -= t * cache[tup]
    return K
