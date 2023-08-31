import itertools
import random

import networkx as nx
import numpy as np

from trilearn.graph import junction_tree as libj, junction_tree as jtlib


def separators(graph):
    """ Returns the separators of graph.

    Args:
        graph (NetworkX graph): A decomposable graph.

    Returns:
        dict: A dict with separators as keys an their corresponding edges as values. 

    Example:
        >>> g = dlib.sample_random_AR_graph(5,3)
        >>> g.nodes
        NodeView((0, 1, 2, 3, 4))
        >>> g.edges
        EdgeView([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)])
        >>>  dlib.separators(g)
        {frozenset([2]): set([(frozenset([2, 3]), frozenset([0, 1, 2]))]), frozenset([3]): set([(frozenset([2, 3]), frozenset([3, 4]))])}

    """
    tree = junction_tree(graph)
    return libj.separators(tree)


def n_junction_trees(graph):
    """Count then number of junctino trees for graph.

    Args:
        graph (NetworkX graph): A decomposable graph.

    Returns:
        int: Number of junction trees for graph.

    Example:
        >>> g = dlib.sample_random_AR_graph(5,3)
        >>> g.nodes
        NodeView((0, 1, 2, 3, 4))
        >>> g.edges
        EdgeView([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)])
        >>>  dlib.separators(g)
        {frozenset([2]): set([(frozenset([2, 3]), frozenset([0, 1, 2]))]), frozenset([3]): set([(frozenset([2, 3]), frozenset([3, 4]))])}
        >>> lib.n_junction_trees(g)
        1.0
    """
    tree = junction_tree(graph)
    seps = separators(graph)
    return np.exp(libj.log_n_junction_trees(tree, seps))


def all_dec_graphs(p):
    """ Returns all decomposable graphs with p nodes  [1]_.

    Args:
        p (int): order of the graphs.

    Returns:
        list: all decomposable graphs with p nodes.

    Note:
        This will only work for small numbers of p (p~10).

    Example:
        >>> glist = dlib.all_dec_graphs(3)
        >>> for graph in glist: 
        ...     print(graph.nodes)
        ...     print(graph.edges)
        ... 
        [0, 1, 2]
        []
        [0, 1, 2]
        [(0, 1), (0, 2)]
        [0, 1, 2]
        [(0, 2)]
        [0, 1, 2]
        [(0, 2), (1, 2)]
        [0, 1, 2]
        [(0, 1)]
        [0, 1, 2]
        [(0, 1), (1, 2)]
        [0, 1, 2]
        [(1, 2)]
        [0, 1, 2]
        [(0, 1), (0, 2), (1, 2)]

    References:
        .. [1] En referens..

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

    Example:
        >>> from trilearn.graph import decomposable as dlib
        >>> g = dlib.naive_decomposable_graph(5)
        >>> dlib.peo(g)
        ([frozenset([1, 2]), frozenset([1, 3, 4]), frozenset([0, 1, 3])], [None, frozenset([1]), frozenset([1, 3])], [frozenset([1, 2]), frozenset([1, 2, 3, 4]), frozenset([0, 1, 2, 3, 4])], [frozenset([2]), frozenset([2, 4])], [frozenset([1, 2]), frozenset([3, 4]), frozenset([0])])
    """

    T = junction_tree(graph)
    return libj.peo(T)


def naive_decomposable_graph(n):
    """ Naive implementation for generating a random decomposable graph.
    Note that this will only for for small numbers (~10) of n.

    Args:
        n (int): order of the samples graph.

    Returns:
        NetworkX graph: A decomposable graph.

    Example:
        >>> g = dlib.naive_decomposable_graph(4)
        >>> g.edges
        EdgeView([(0, 1), (0, 3)])
        >>> g.nodes
        NodeView((0, 1, 2, 3))
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
    """ Generates a random decomposable graph using the Christmas tree algorithm.

    Args:
        internal_nodes (list): list of internal nodes in the generated graph.
        alpha (float): Subtree kernel parameter
        beta (float): Subtree kernel parameter
        directory (string): Path to where the plots should be saved.

    Returns:
        NetworkX graph: A decomposable graph.

    Example:
        >>> g = dlib.sample_dec_graph(5)
        >>> g.edges
        EdgeView([(0, 1), (0, 3), (1, 3), (2, 3)])
        >>> g.nodes
        NodeView((0, 1, 2, 3, 4))

    """

    T = libj.sample(internal_nodes, alpha=alpha, beta=beta)
    return libj.graph(T)


def sample(order, alpha=0.5, beta=0.5):
    """ Generates a random decomposable graph using the Christmas tree algorithm.

    Args:
        internal_nodes (list): list of internal nodes in the generated graph.
        alpha (float): Subtree kernel parameter
        beta (float): Subtree kernel parameter
        directory (string): Path to where the plots should be saved.

    Returns:
        NetworkX graph: A decomposable graph.

    Example:
        >>> g = dlib.sample_dec_graph(5)
        >>> g.edges
        EdgeView([(0, 1), (0, 3), (1, 3), (2, 3)])
        >>> g.nodes
        NodeView((0, 1, 2, 3, 4))

    """

    if type(order) is int:
        nodes = range(order)  # OBS. Python 2.7
        random.shuffle(nodes)
        tree = libj.sample(nodes, alpha, beta)
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

    Example:
        >>> g = dlib.sample_dec_graph(5)
        >>> t = dlib.junction_tree(g)
        >>> t.nodes
        NodeView((frozenset([4]), frozenset([2, 3]), frozenset([0, 1, 3])))
        >>> t.edges
        EdgeView([(frozenset([4]), frozenset([2, 3])), (frozenset([2, 3]), frozenset([0, 1, 3]))])
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


def gen_AR_graph(n_dim, width=2):
    """Generates a graph with k-band adjacency matrix

    Args:
        n_dim (NetworkX graph): Number of nodes.

    Returns:
        NetworkX graph: A graph with k-band adjacency matrix. 
        Can represent depdndence structure in a AR2 model.

    Example:
        >>> g = dlib.gen_AR_graph(5)
        >>> g.nodes
        NodeView((0, 1, 2, 3, 4))
        >>> g.edges
        EdgeView([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)])
    """
    m = np.eye(n_dim, k=0, dtype=int)
    for i in range(1, width+1):
        m += np.eye(n_dim, k=i,dtype=int)
        m += np.eye(n_dim, k=-i,dtype=int)
    graph = nx.from_numpy_matrix(m)

    return graph


def sample_random_AR_graph(n_dim, max_bandwidth):
    """ Sample graph with adjancency matrix with varying bandwidth. 

    Args:
        n_dim (int): number of nodes.
        max_bandwidth (int): Maximum band width.

    Returns:
        Networkx graph: A graph with band adjancency matrix. 

    Example:
        >>> g = dlib.sample_random_AR_graph(5,3)
        >>> g.nodes
        NodeView((0, 1, 2, 3, 4))
        >>> g.edges
        EdgeView([(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)])
    """
    adjmat = np.zeros(n_dim*n_dim, dtype=int).reshape((n_dim, n_dim))
    bandwidth = 1
    for i in range(n_dim):
        b = np.random.choice([-1, 0, 1], 1, p=np.array([1, 1, 1])/3.0)[0]
        bandwidth = max(bandwidth + b, 1)
        bandwidth = min(bandwidth, n_dim - i - 1)
        bandwidth = min(bandwidth, max_bandwidth)

        for j in range(bandwidth):
            adjmat[i, i + j + 1] = 1
            adjmat[i + j + 1, i] = 1

    graph = nx.from_numpy_array(adjmat)

    return graph
