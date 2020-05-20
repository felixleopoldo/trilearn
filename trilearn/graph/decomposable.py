import itertools
import random

import networkx as nx
import numpy as np

from trilearn.graph import junction_tree as libj, junction_tree as jtlib
from trilearn.graph import almond_tree as atlib

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
    """ Generates a random decomposable graph using the Christmas tree algorithm.

    Args:
        internal_nodes (list): list of internal nodes in the generated graph.
        alpha (float): Subtree kernel parameter
        beta (float): Subtree kernel parameter
        directory (string): Path to where the plots should be saved.

    Returns:
        NetworkX graph: a decomposable graph.
    """

    T = libj.sample(internal_nodes, alpha=alpha, beta=beta)
    return libj.graph(T)


def sample(order, alpha=0.5, beta=0.5):

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
    jt.num_graph_nodes = len(graph)
    return jt

def almond_tree(graph):
    """ Returns the unique almond tree representation of graph.

    Args:
        graph (NetworkX graph): A decomposable graph

    Returns:
        NetworkX graph: An almond tree
    """
    jt = junction_tree(graph)
    almondt = atlib.AlmondTree()
    seps = jt.get_separators()
    almondt.add_nodes_from(jt.nodes())
    almondt.add_separators_from(seps.keys())
    clique_sep_edges = []
    for s, e in seps.items():
        for x in e:
            l = list(x)
            clique_sep_edges.append((s, l[0]))
            clique_sep_edges.append((s, l[1]))
    sep_sep_edges = []
    for n1 in seps.keys():
        for n2 in seps.keys():
            if n1 < n2 and n1 != frozenset([]):
                sep_sep_edges.append((n1, n2))
    
    almondt.add_edges_from(clique_sep_edges, weight=0)
    almondt.add_edges_from(sep_sep_edges, weight=0)
    # implementation from Jensen (1994) Optimal Junction tree, and
    # Almond (1993) Optimality issues in constructing a Markov tree from Graphical Models
    for s in seps.keys():
        multi = len(seps[s]) + 1
        n_edges_to_remove = almondt.degree(s) - multi
        if n_edges_to_remove > 0:
            nei_C = almondt.neighbors_cliques(s)
            nei_S = almondt.neighbors_separators(s)
            for n in nei_S:
                nei_n_C = almondt.neighbors_cliques(n)
                inter = list(set(nei_n_C) & set(nei_C))
                if s < n and inter:
                    for x in inter:
                        if almondt.has_edge(x, s):
                            almondt[x][s]['weight'] -= 1
                            almondt[n][x]['weight'] -= 1

    T = nx.minimum_spanning_tree(almondt)
    at = atlib.AlmondTree()
    at.add_nodes_from(jt.nodes())
    at.add_separators_from(seps.keys())
    at.add_edges_from(T.edges())

    # testing number of degrees = multiplicity + 1
    for sp in seps:
        if at.degree(sp) != len(seps[sp]) + 1:
            print('{} ({}, {})'.format(sp, at.degree(sp), len(seps[sp])+1))
            import pdb; pdb.set_trace()

    return at


def gen_AR2_graph(n_dim):
    graph = nx.Graph()
    for i in range(n_dim):
        graph.add_node(i, label=str(i+1))
    for i in range(n_dim-2):
        graph.add_edges_from([(i, i+1), (i, i+2)])
    graph.add_edge(n_dim-2, n_dim-1)
    return graph


def sample_random_AR_graph(n_dim, max_bandwidth):
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

    graph = nx.from_numpy_matrix(adjmat)

    return graph
