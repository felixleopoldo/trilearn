import math

import numpy as np

import sys

def log_norm_constant(alpha):
    """ Log of the normalizing constant in the Dirichlet distribution.

    Args:
        alpha (numpy array float): alpha vector.

    Returns:
        float: the normalizing constant.
    """
    tmp = np.sum([math.lgamma(a) for a in alpha])
    tmp -= math.lgamma(np.sum(alpha))
    return tmp


def log_pdf(x, alpha):
    """ Log density function of the Dirichlet distribution.

    Args:
        alpha (np.array float): alpha vector.
        x (float): function argument.

    Returns:
        float: the normalizing constant.

    """
    tmp = np.inner(alpha-1, np.log(x))  # inner product
    tmp = tmp - log_norm_constant(alpha)
    return tmp


def pdf(x, alpha):
    """ The density function f(x) of a one-dimensional Dirichlet distribution.

    Args:
        alpha (np.array float): alpha vector.
        x (float): function argument.

    Returns:
        float: the normalizing constant.
    """
    return np.exp(log_pdf(x, alpha))


def pdf_multidim(x, alpha, beta, levels):
    """ The density function f(x) of a multi-dimensional Dirichlet distribution.

    Args:
        alpha (dict): a dictionary specifying specific pseudo counts.
        beta (float): is added to all pseudo counts. This is to avoid storing large pseudo count tables.

    Returns:
        float: the density at x, f(x)
    """
    pass


def log_norm_constant_multidim(alpha, beta, levels):
    """
    The normalizing constant in a multidimensional multinomial distribution.

    Args:
        levels (list): A list of the number of level for each random variable. e.g. [2, 2, 3].
        alpha (dict): A dictiory of cells and the pseudo counts added for that cell.
        beta (float): A constant pseudo count for each cell.

    Returns:
        float: the normalizing constant.
    """
    no_cells = np.prod([len(l) for l in levels])
    numerator = (no_cells-len(alpha)) * math.lgamma(beta)

    #for cell, count in alpha.iteritems():
    for cell, count in alpha.items():
        numerator += math.lgamma(count + beta)
    #n = np.sum([val for key, val in alpha.iteritems()])
    n = np.sum([val for key, val in alpha.items()])
    denominator = math.lgamma(no_cells*beta + n)

    return numerator - denominator
