import itertools

import networkx as nx
import numpy as np

from trilearn.graph import junction_tree as libj, junction_tree as jtlib


def separators(graph):
    """ Returns the separators of graph.

    Args:
        graph (NetworkX graph): A decomposable graph

    Returns:
        dict:  Example {sep1: [sep1_edge1, sep1_edge2, ...], sep2: [...]}
    """
    tree = junction_tree(graph)
    return libj.separators(tree)


def n_junction_trees(graph):
    tree = junction_tree(graph)
    seps = separators(graph)
    return np.exp(libj.log_n_junction_trees(tree, seps))


def all_dec_graphs(p):
    """ Returns all decomposable graphs with p nodes.

    Args:
        p (int): order of the graphs.

    Returns:
        list: all decomposable graphs with p nodes.

    Note:
        This will only work for small numbers of p.
    """
    graphs = set()
    for val in itertools.product(*([[0, 1]] * p**2)):
        mat = np.array(val).reshape(p, p)
        if np.all(mat == mat.T) and np.all(mat.diagonal() == np.zeros(p)):
            graph = nx.from_numpy_matrix(mat)
            if nx.is_chordal(graph):
                graphs.add(graph)
    return graphs


def peo(graph):
    """ Returns a perfect elimination order of graph.

    Args:
        graph (NetworkX graph): a decomposable graph.

    Returns:
        a perfect elimination order of graph.
    """

    T = junction_tree(graph)
    return libj.peo(T)


def naive_decomposable_graph(n):
    """ Naive implementation for generating a random decomposable graph.
    Note that this will only for for small numbers (~10) of n.

    Args:
        n (int): order of the samples graph

    Returns:
        NetworkX graph: a decopmosable graph
    """
    m = np.zeros(n*n, dtype=int)
    m.resize(n, n)
    for i in range(n-1):
        for j in range(i+1, n):
            e = np.random.randint(2)
            m[i, j] = e
            m[j, i] = e
    graph = nx.from_numpy_matrix(m)

    while not nx.is_chordal(graph):
        for i in range(n-1):
            for j in range(i+1, n):
                e = np.random.randint(2)
                m[i, j] = e
                m[j, i] = e
        graph = nx.from_numpy_matrix(m)

    return graph


def sample_dec_graph(internal_nodes, alpha=0.5, beta=0.5, directory='.'):
    """ Generates a random decomposable graph using the Christmas tree algotihm.

    Args:
        internal_nodes (list): list of internal nodes in the generated graph.
        alpha (float): Subtree kernel parameter
        beta (float): Subtree kernel parameter
        directory (string): Path to where the plots should be saved.

    Returns:
        NetworkX graph: a decomposable graph.
    """
    T = libj.gen_JT(internal_nodes, alpha, beta, directory)
    return libj.graph(T)


def sample_decomposable_graph(order, alpha=0.5, beta=0.5):
    if type(order) is int:
        tree = libj.sample(range(order), alpha, beta)
        return jtlib.graph(tree)
    elif type(order) is list:
        tree = libj.sample(order, alpha, beta)
        return jtlib.graph(tree)


def junction_tree(graph):
    """ Returns a junction tree representation of graph.

    Args:
        graph (NetworkX graph): A decomposable graph

    Returns:
        NetworkX graph: A junction tree.
    """
    CG = nx.Graph()
    for c in nx.find_cliques(graph):
        CG.add_node(frozenset(c), label=str(tuple(c)), color="red")
    for c1 in CG.nodes():
        for c2 in CG.nodes():
            if not CG.has_edge(c1, c2) and not c1 == c2:
                lab = str(tuple(c1.intersection(c2)))
                if len(tuple(c1.intersection(c2))) == 1:
                    lab = "(" + str(list(c1.intersection(c2))[0]) + ")"
                CG.add_edge(c1, c2, weight=-len(c1.intersection(c2)),
                            label=lab)
    T = nx.minimum_spanning_tree(CG)
    jt = libj.JunctionTree()
    jt.add_nodes_from(T.nodes())
    jt.add_edges_from(T.edges())
    return jt