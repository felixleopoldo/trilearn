"""
Graph related functions.
"""

import itertools
import json

from networkx.readwrite import json_graph
import networkx as nx
import numpy as np
import seaborn as sns
from scipy.special import comb

import trilearn.auxiliary_functions


def from_json_file(filename):
    """From json graph to graph.

    Args:
        filename (string): Filename of json graph.

    Returns:
        NetworksX graph: NetworkX version of the json graph.
    """
    with open(filename) as data_file:
        json_G = json.load(data_file)

    return json_graph.node_link_graph(json_G)


def replace_node(graph, node, new_node):
    """Replaces node by new_node in graph.

    Args:
        graph (NetworkX graph): A graph.
        node (hashable object): A node.
        new_node (hashable object): Another node.
    """
    graph.add_node(new_node)
    graph.add_edges_from([(new_node, n) for n in graph.neighbors(node)])
    graph.remove_node(node)


def plot(graph, filename, layout="dot"):
    """ Plots a networkx graph and saves it to filename.

    Args:
        graph (NetworkX graph): A graph.
        filename (string): The filename.

    """
    agraph = nx.nx_agraph.to_agraph(graph)
    agraph.layout(layout)
    agraph.draw(filename)


def graph_to_tuple(graph):
    """ Takes a NetworkX graph and returns a tuplized adjacency matrix.

    Args:
        graph (NetworkX graph): A graph

    Returns:
        tuple: A flattened adjacency matrix in tuple format.
    
    Example:
        >>> g.nodes
        NodeView((0, 1, 2, 3, 4))
        >>> g.edges
        EdgeView([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)])
        >>> glib.graph_to_tuple(g)
        (0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0)

    """
    p = graph.order()
    mat = nx.to_numpy_array(graph, dtype=int, nodelist=range(p)).reshape(p*p)    
    return tuple(mat)


def tuple_to_graph(vecmat):
    """ Takes a tuple of the rows in an adjacency matrix
    and returns a nx.graph. This is a kind of serialization of a graph.

    Args:
        vecmat (tuple): tuple of the rows in an adjacency matrix.

    Returns:
        NetworkX graph
    """
    p = int(np.sqrt(len(vecmat)))

    mat = np.array(vecmat).reshape(p, p)    
    mat += mat.T
    return nx.from_numpy_array(mat)


def hash_graph(graph):
    """ A hash value of the tupelized version of graph.

    Args:
        graph (NetworkX graph): A graph

    Returns:
        int: A hash value of a graph.

    Example:
        >>> g = dlib.sample(5)
        >>> g.nodes
        NodeView((0, 1, 2, 3, 4))
        >>> g.edges
        EdgeView([(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)])   
        >>> glib.hash_graph(g)
        249771633555694270
    """
    return hash(str(graph_to_tuple(graph)))


def true_distribution(seqdist, filename):
    """Calculating true distribution for a graph with 6 nodes.

    Args:
        seqdist (SequentialDistribution): A (Sequential) distribution for a decomposable graph.
        filename (string): Filename to save marginal edge distribtion.

    Returns:
        dict: The graph distribution evaluated for each graph.
    """
    p = seqdist.p
    no_chordal = 0
    true_heatmap = np.matrix(np.zeros(p*p).reshape(p, p))
    max_ll = -100000
    graph_ll = {}
    graph_post = {}
    for val in itertools.product(*([[0, 1]] * comb(p, 2))):
        vec_mat = [0]
        vec_mat += list(val[0:5])
        vec_mat += [0]*2
        vec_mat += list(val[5:9])
        vec_mat += [0]*3
        vec_mat += list(val[9:12])
        vec_mat += [0]*4
        vec_mat += list(val[12:14])
        vec_mat += [0]*5
        vec_mat += [val[14]]
        vec_mat += [0]*6
        mat = np.array(vec_mat).reshape(p, p)
        mat += mat.T
        mat = np.matrix(mat)
        graph1 = nx.from_numpy_matrix(mat)

        if nx.is_chordal(graph1):
            no_chordal += 1
            logl = seqdist.log_likelihood(graph1)
            if logl > max_ll:
                max_ll = logl
            graph_ll[tuple(vec_mat)] = logl
    # Rescaled normalizing constant
    norm_const_rescaled = sum([np.exp(rp-max_ll)
                               for g, rp in graph_ll.iteritems()])

    for vec_mat, ll in graph_ll.iteritems():
        mat = np.array(vec_mat).reshape(p, p)
        mat += mat.T
        mat = np.matrix(mat)
        graph1 = nx.from_numpy_matrix(mat)
        if nx.is_chordal(graph1):
            graph_post[vec_mat] = np.exp(ll-max_ll) / norm_const_rescaled
            true_heatmap += mat * graph_post[vec_mat]

        with sns.axes_style("white"):
            sns.heatmap(heatmap, mask=mask, annot=False,
                        cmap="Blues",
                        xticklabels=range(1, p+1),
                        yticklabels=range(1, p+1),
                        vmin=0.0, vmax=1.0, square=True,
                        cbar=True)
            plt.yticks(rotation=0)

            plt.savefig(filename_prefix+"_edge_heatmap_cbar.eps",
                        format="eps",bbox_inches='tight', dpi=100)
            plt.clf()

    trilearn.auxiliary_functions.plot_matrix(np.array(true_heatmap), filename, "png",
                                             title="Czech Autoworkers posterior heatmap, lambda=" +
                   str(seqdist.cell_alpha))
    return graph_post


def plot_adjmat(graph, cbar=False):
    """ Plots the adjecency matrix of graph.

    Args:
        graph (NetworkX graph): A graph.
    """
    heatmap = nx.to_numpy_matrix(graph)
    mask = np.zeros_like(heatmap)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(heatmap, mask=mask, annot=False,
                    cmap="Blues",
                    vmin=0.0, vmax=1.0, square=True,
                    cbar=cbar, xticklabels=5, yticklabels=5)
