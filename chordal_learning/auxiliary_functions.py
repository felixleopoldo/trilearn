import numpy as np
from numpy import linalg as la
from matplotlib import pyplot as plt


def random_subset(A):
    """ Draws a random subset of elements in a list, inclding the empty set.

    Args:
        A (list)

    Returns:
        set: Subset of A.
    """
    tmp = np.array(list(A))
    bin_samp = np.random.multinomial(1,  [0.5, 0.5], size=len(tmp))
    c = np.ma.masked_array(tmp, mask=bin_samp[:, 0])
    rest = set(c.compressed())
    return rest


def random_element_from_coll(A):
    tmp = np.array(list(A))
    ind = np.random.randint(len(A))
    return tmp[ind]


def l2_loss(m1, m2):
    """ L2 loss between m1 and m2.

    Args:
        m1 (Numpy array): A matrix
        m1 (Numpy array): A matrix

    Returns:
        float
    """
    A = np.matrix(m1)
    B = np.matrix(m2)
    return np.power(la.norm(A - B, "fro"), 2)  # <A-B, A-B>


def l1_loss(m1, m2):
    """ L1 loss.

    Args:
        m1 (Numpy array): A matrix
        m1 (Numpy array): A matrix

    Returns:
        float
    """
    A = np.matrix(m1)
    B = np.matrix(m2)
    p = A.shape[0]
    (sign, logdet) = la.slogdet(A * B.I)
    loss = np.trace(A.transpose() * B.I)  # <A, B.I>
    loss -= sign * logdet
    loss -= p
    return loss


def tpr(true_graph, est_graph):
    """ Calculates the True positive rate of an estimated adjacency matrix.
    """
    N = len(true_graph)
    no_correct = 0.0
    no_false_rej = 0.0

    for i in range(N):
        for j in range(N):
            if est_graph.item(i, j) == 1 and true_graph.item(i, j) == 1:
                    no_correct += 1

            if true_graph.item(i, j) == 1:
                if est_graph.item(i, j) == 0:
                    no_false_rej += 1
    return no_correct / (no_correct + no_false_rej)


def spc1(true_graph, est_graph):
    """
    Takes 2 adjacency matrices.
    """
    N = len(true_graph)
    no_corr_rej = 0.0
    no_wrong_incl = 0.0

    for i in range(N):
        for j in range(N):
            if est_graph.item(i, j) == 1 and true_graph.item(i, j) == 0:
                no_wrong_incl += 1
            if est_graph.item(i, j) == 0 and true_graph.item(i, j) == 0:
                no_corr_rej += 1

    return no_corr_rej / (no_corr_rej + no_wrong_incl)


def get_marg_counts(full_data, subset):
    """ Returns a contingency table in dictionary form.

    Args:
        data (np.array): The data in n x p form.
        subset (list): The subset of interest
    """
    if len(subset) == 0:
        return None
    counts = {}
    for row in full_data:
        cell = tuple(row[subset])
        # print cell
        if cell not in counts:
            counts[cell] = 1
        else:
            counts[cell] += 1
    return counts


def plot_matrix(m, filename, extension, title="Adjmat"):
    """ Plots a 2-dim numpy array as heatmap.
        Args:
            m (numpy array): matrix to plot.
    """
    m1 = np.array(m)
    fig, ax = plt.subplots()
    ax.pcolor(m1, cmap=plt.cm.Blues)
    fig.suptitle(title, fontsize=20)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    plt.savefig(filename+"."+extension, format=extension, dpi=100)