import numpy as np
import pandas as pd
from numpy import linalg as la
from matplotlib import pyplot as plt
import seaborn as sns


def plot_heatmap(heatmap, cbar=False, annot=False):
    mask = np.zeros_like(heatmap)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(heatmap, mask=mask, annot=annot,
                    cmap="Blues",
                    vmin=0.0, vmax=1.0, square=True,
                    cbar=cbar, xticklabels=5, yticklabels=5)


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


def sample_classification_datasets(mus, covmats, n_samples_in_each_class):
    n_classes = len(covmats)
    n_dim = covmats[0].shape[1]
    # Generate training data
    n_train = [n_samples_in_each_class] * n_classes
    x = np.matrix(np.zeros((sum(n_train), n_dim))).reshape(sum(n_train), n_dim)
    y = np.matrix(np.zeros((sum(n_train), 1), dtype=int)).reshape(sum(n_train), 1)

    for c in range(n_classes):
        fr = sum(n_train[:c])
        to = sum(n_train[:c + 1])
        x[np.ix_(range(fr, to),
                       range(n_dim))] = np.matrix(np.random.multivariate_normal(
                                                  np.array(mus[c]).flatten(),
                                                  covmats[c],
                                                  n_train[c]))
        y[np.ix_(range(fr, to), [0])] = np.matrix(np.ones(n_train[c], dtype=int) * c).T

    ys = pd.Series(np.array(y).flatten(), dtype=int)
    df = pd.DataFrame(x)
    df["y"] = ys
    df = df[["y"] + range(n_dim)]
    df.columns = ["y"] + ["x" + str(i) for i in range(1, n_dim + 1)]
    return df
    # return x, y
    # return pd.DataFrame(x), pd.Series(np.array(y).flatten(), dtype=int)