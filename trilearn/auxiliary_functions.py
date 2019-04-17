import glob

import networkx as nx
import numpy as np
import pandas as pd
from numpy import linalg as la
from matplotlib import pyplot as plt
import seaborn as sns
from pandas.plotting import autocorrelation_plot


def plot_heatmap(heatmap, cbar=False, annot=False, xticklabels=1, yticklabels=1):
    mask = np.zeros_like(heatmap)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(heatmap, mask=mask, annot=annot,
                    cmap="Blues",
                    vmin=0.0, vmax=1.0, square=True,
                    cbar=cbar,
                    xticklabels=xticklabels, yticklabels=yticklabels)


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


def plot_multiple_traj_statistics(trajectories, burnin_end,
                                  write_to_file=False, output_directory="./", file_extension="eps"):
    print(trajectories)

    for param_setting, traj_list in trajectories.iteritems():
        print(traj_list[0].sampling_method)
        print("Average sample time: " + str(np.mean(traj_list[0].time)))

        print("Plotting size")
        sns.set_style("whitegrid")
        for t in traj_list:
            t.size(burnin_end).plot()
        if write_to_file:
            plt.savefig(output_directory +"/"+ str(t) + "_size." + file_extension)
        plt.clf()

        print("Plotting log-likelihood")
        sns.set_style("whitegrid")
        for t in traj_list:
            t.log_likelihood(burnin_end).plot()
        if write_to_file:
            plt.savefig(output_directory +"/"+ str(t) + "_log-likelihood."+file_extension)
        plt.clf()

        print("Plotting autocorr")
        sns.set_style("whitegrid")
        for t in traj_list:
            autocorrelation_plot(t.size(burnin_end))
        if write_to_file:
            plt.savefig(output_directory +"/"+ str(t) + "_autocorr."+file_extension)
        plt.clf()


        print("Plotting heatmap, size auto-correlation, MAP and ML graph")
        for i, t in enumerate(traj_list):
            plot_heatmap(t.empirical_distribution(burnin_end).heatmap(),
                         xticklabels=np.arange(1, t.seqdist.p +1),
                         yticklabels=np.arange(1, t.seqdist.p +1))
            cax = plt.gcf().axes[-1]
            cax.tick_params(labelsize=6)
            if write_to_file:
                plt.savefig(output_directory +"/"+ str(t) + "_heatmap_" + str(i) + "."+file_extension)
            plt.clf()

            plot_heatmap(t.empirical_distribution(burnin_end).heatmap(), cbar=True,
                         xticklabels=np.arange(1, t.seqdist.p +1),
                         yticklabels=np.arange(1, t.seqdist.p +1))
            cax = plt.gcf().axes[-1]
            cax.tick_params(labelsize=6)
            cax = plt.gcf().axes[-2]
            cax.tick_params(labelsize=6)

            if write_to_file:
                plt.savefig(output_directory +"/"+ str(t) + "_heatmap_cbar_" + str(i) + "."+file_extension)
            plt.clf()

            sns.set_style("white")
            autocorrelation_plot(t.size(burnin_end))
            if write_to_file:
                plt.savefig(output_directory +"/"+ str(t) + "_size_autocorr_" + str(i) + "."+file_extension)
            plt.clf()

            top = t.empirical_distribution().mode(1)
            plot_heatmap(nx.to_numpy_array(top[0][0]))
            cax = plt.gcf().axes[-1]
            cax.tick_params(labelsize=6)
            if write_to_file:
                plt.savefig(output_directory +"/"+ str(t) + "_map_" + str(i) + "."+file_extension)
            plt.clf()

            plot_heatmap(nx.to_numpy_array(t.maximum_likelihood_graph()))
            cax = plt.gcf().axes[-1]
            cax.tick_params(labelsize=6)
            if write_to_file:
                plt.savefig(output_directory +"/"+ str(t) + "_ml_" + str(i) + "."+file_extension)
            plt.clf()

def read_all_trajectories_in_dir(directory):
    trajectories = {}
    from trilearn.graph import trajectory as gtraj
    for filename in glob.glob(directory + "/*.json"):
        print("Loading: " + str(filename))
        # Gather all with the same parameter setting in the same plot
        t = gtraj.Trajectory()
        t.read_file(filename)
        if str(t) not in trajectories:
            trajectories[str(t)] = []
        trajectories[str(t)].append(t)
    return trajectories


def plot_graph_traj_statistics(graph_traj, write_to_file=False):
    top = graph_traj.empirical_distribution().mode(5)
    print("Probability\tEdge list: ")
    for graph, prob in top:
        print(str(prob) + "\t\t" + str(list(graph.edges())))

    graph_traj.size().plot()
    if write_to_file:
        plt.savefig(str(graph_traj)+"_size.png")
    plt.clf()

    autocorrelation_plot(graph_traj.size())
    if write_to_file:
        plt.savefig(str(graph_traj)+"_autocorr.png")
    plt.clf()

    graph_traj.log_likelihood().plot()
    if write_to_file:
        plt.savefig(str(graph_traj)+"_loglik.png")
    plt.clf()

    plot_heatmap(graph_traj.empirical_distribution().heatmap())
    if write_to_file:
        plt.savefig(str(graph_traj)+"_heatmap.png")
    plt.clf()

    top = graph_traj.empirical_distribution().mode(1)
    plot_heatmap(nx.to_numpy_array(top[0][0]))
    if write_to_file:
        plt.savefig(str(graph_traj)+"_map.png")
    plt.clf()