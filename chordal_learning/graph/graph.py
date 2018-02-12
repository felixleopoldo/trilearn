"""
This library provides graph functions.
"""

from collections import deque
import itertools

import numpy as np
import networkx as nx
import seaborn as sns
from scipy.special import comb

import chordal_learning.auxiliary_functions
import chordal_learning.graph.junction_tree as libj


def plot(graph, filename, layout="dot"):
    """ Plots a networkx graph and saves it to filename.

    Args:
        graph (NetworkX graph): A graph
        filename (string): The filename

    """
    agraph = nx.nx_agraph.to_agraph(graph)
    agraph.layout(layout)
    agraph.draw(filename)


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


def graph_to_tuple(graph):
    """ Takes a NetworkX graph and returns a tuplized adjacency matrix.

    Args:
        graph (NetworkX graph): A graph

    Returns:
        tuple: E.g. (1,0,0,...,0,0,1,0)

    """
    p = graph.order()
    mat = np.array(nx.to_numpy_matrix(graph), dtype=int).reshape(p*p)
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
    mat = np.matrix(mat)
    return nx.from_numpy_matrix(mat)


def hash_graph(graph):
    """ A hash value of the tupelized version of graph.

    Args:
        graph (NetworkX graph): A graph

    Returns:
        int
    """
    return hash(str(graph_to_tuple(graph)))


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


def true_distribution(seqdist, filename):
    """ Calculating true distribution for a graph with 6 nodes.
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
            logl = seqdist.ll(graph1)
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

    chordal_learning.auxiliary_functions.plot_matrix(np.array(true_heatmap), filename, "png",
                                                     title="Czech Autoworkers posterior heatmap, lambda=" +
                   str(seqdist.cell_alpha))
    return graph_post


def random_subtree(T, alpha, beta, subtree_mark):
    """ Random subtree of T according to Algorithm X in [1].

    Args:
        alpha (float): probability of continuing to a neighbor
        beta (float): probability of non empty subtree
        T (NetworkX graph): the tree of which the subtree is taken

    Returns:
        A subtree of T

    References:
        [1] F. Rios J., Ohlsson, T. Pavlenko Bayesian structure learning in graphical models using sequential Monte Carlo.

    """
    # Take empty subtree with prob beta
    empty = np.random.multinomial(1, [beta, 1-beta]).argmax()
    subtree_edges = []
    subtree_nodes = []

    if empty == 1:
        separators = {}
        subtree = nx.Graph()
        return (subtree, [], [], {}, separators, 1-beta)

    # Take non-empty subtree
    n = T.order()
    w = 0.0
    visited = set()  # cliques
    q = deque([])
    start = np.random.randint(n)  # then n means new component
    separators = {}
    start_node = T.nodes()[start]
    q.append(start_node)
    subtree_adjlist = {start_node: []}
    while len(q) > 0:
        node = q.popleft()
        visited.add(node)
        subtree_nodes.append(node)
        T.node[node]["subnode"] = subtree_mark
        for neig in T.neighbors(node):            
            b = np.random.multinomial(1, [1-alpha, alpha]).argmax()
            if neig not in visited:
                if b == 1:
                    subtree_edges.append((node, neig))
                    subtree_adjlist[node].append(neig)
                    subtree_adjlist[neig] = [node]
                    q.append(neig)
                    # Add separator
                    sep = neig & node
                    if not sep in separators:
                        separators[sep] = []
                    separators[sep].append((neig, node))
                else:
                    w += 1

    subtree = T.subgraph(subtree_nodes)
    v = len(subtree_nodes)
    probtree = beta * v * np.power(alpha, v-1) / np.float(n)
    probtree *= np.power(1-alpha, w)
    return (subtree, subtree_nodes, subtree_edges, subtree_adjlist, separators, probtree)


def peo(graph):
    """ Returns a perfect elimination order of graph.

    Args:
        graph (NetworkX graph): a decomposable graph.

    Returns:
        a perfect elimination order of graph.
    """

    T = junction_tree(graph)
    return libj.peo(T)


def prob_subtree(subtree, T, alpha, beta):
    """ Returns the probability of the subtree subtree generated by
    random_subtree(T, alpha, beta).

    Args:
        T (NetworkX graph): A tree
        subtree (NetworkX graph): a subtree of T drawn by the subtree kernel
        alpha (float): Subtree kernel parameter
        beta (float): Subtree kernel parameter

    Returns:
        float
    """
    p = subtree.order()
    if p == 0:
        return 1.0 - beta
    forest = T.subgraph(set(T.nodes()) - set(subtree.nodes()))
    components = nx.connected_components(forest)
    w = float(len(list(components)))
    v = float(subtree.order())
    alpha = float(alpha)
    beta = float(beta)
    n = float(T.order())
    prob = beta * v * np.power(alpha, v-1) * np.power(1-alpha, w) / n
    return prob


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


def plot_adjmat(graph, cbar=False):
    """ Plots the adjecency matrix of graph.

    Args:
        graph (NetworkX graph): a graph
    """
    heatmap = nx.to_numpy_matrix(graph)
    mask = np.zeros_like(heatmap)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        sns.heatmap(heatmap, mask=mask, annot=False,
                    cmap="Blues",
                    vmin=0.0, vmax=1.0, square=True,
                    cbar=cbar, xticklabels=5, yticklabels=5)


def sample_dec_graph(internal_nodes, alpha=0.5, beta=0.5, directory='.'):
    """ Generates a random decomposable graph using the Christmas tree algotihm.

    Args:
        internal_nodes (list): list of inernal nodes in the generated graph.
        alpha (float): Subtree kernel parameter
        beta (float): Subtree kernel parameter
        directory (string): Path to where the plots should be saved.

    Returns:
        NetworkX graph: a decomposable graph.
    """
    T = libj.gen_JT(internal_nodes, alpha, beta, directory)
    return libj.graph(T)


def to_prufer(tree):
    """ Generate Prufer sequence for tree.

    Args:
        tree (NetwokrX.Graph): a tree.

    Returns:
        list: the Prufer sequence.
    """
    graph = tree.subgraph(tree.nodes())
    if not nx.is_tree(graph):
        return False
    order = graph.order()
    prufer = []
    for _ in range(order-2):
        leafs = [(n, graph.neighbors(n)[0]) for n in graph.nodes() if len(graph.neighbors(n)) == 1]
        leafs.sort()
        prufer.append(leafs[0][1])
        graph.remove_node(leafs[0][0])

    return prufer


def from_prufer(a):
    """
    Prufer sequence to tree
    """
    # n = len(a)
    # T = nx.Graph()
    # T.add_nodes_from(range(1, n+2+1))  # Add extra nodes
    # degree = {n: 0 for n in range(1, n+2+1)}
    # for i in T.nodes():
    #     degree[i] = 1
    # for i in a:
    #     degree[i] += 1
    # for i in a:
    #     for j in T.nodes():
    #         if degree[j] == 1:
    #             T.add_edge(i, j)
    #             degree[i] -= 1
    #             degree[j] -= 1
    #             break
    # print degree
    # u = 0  # last nodes
    # v = 0  # last nodes
    # for i in T.nodes():
    #     if degree[i] == 1:
    #         if u == 0:
    #             u = i
    #         else:
    #             v = i
    #             break
    # T.add_edge(u, v)
    # degree[u] -= 1
    # degree[v] -= 1
    # return T

    n = len(a)
    T = nx.Graph()
    T.add_nodes_from(range(n+2))  # Add extra nodes
    degree = [0 for _ in range(n+2)]
    for i in T.nodes():
        degree[i] = 1
    for i in a:
        degree[i] += 1
    for i in a:
        for j in T.nodes():
            if degree[j] == 1:
                T.add_edge(i, j)
                degree[i] -= 1
                degree[j] -= 1
                break
    u = 0  # last nodes
    v = 0  # last nodes
    for i in T.nodes():
        if degree[i] == 1:
            if u == 0:
                u = i
            else:
                v = i
                break
    T.add_edge(u, v)
    degree[u] -= 1
    degree[v] -= 1
    return T


def jt_to_prufer(tree):
    ind_to_nodes = tree.nodes()
    nodes_to_ind = {ind_to_nodes[i]: i for i in range(tree.order())}
    edges = [(nodes_to_ind[e1], nodes_to_ind[e2]) for (e1, e2) in tree.edges()]
    graph = nx.Graph()
    graph.add_nodes_from(range(tree.order()))
    graph.add_edges_from(edges)
    prufer = to_prufer(graph)



