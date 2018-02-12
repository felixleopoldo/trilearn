"""
Students t-distribution.
"""
import math

import numpy as np
from numpy.linalg import slogdet


def log_pdf(x, mu, T, n):
    [k, N] = x.shape
    xm = x-mu
    (sign, logdet) = slogdet(T)
    logc = math.lgamma((n + k)/2.0)
    logc += 0.5*logdet
    logc -= math.lgamma(n/2.0)
    logc -= np.log(n*math.pi) * (k/2.0)
    logp = -0.5 * (n+k) * np.log(1 + xm.T * T * xm/n)
    logp = logp + logc
    return float(logp)
