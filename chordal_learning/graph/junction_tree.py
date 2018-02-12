"""
Functions related to junction trees.
"""

import networkx as nx
import numpy as np

import chordal_learning.graph as glib


class JunctionTree(nx.Graph):
    ids = {}
    sepdict = {}

    def __init__(self, data=None, **attr):
        nx.Graph.__init__(self, data, **attr)
        self.log_nus = {}
        self.separators = None
        self.altered = True

    def log_nu(self, sep):
        if sep not in self.log_nus:
            self.log_nus[sep] = log_nu(self, sep)
        return self.log_nus[sep]

    def get_separators(self):
        if self.separators is None:
            self.separators = separators(self)
        return self.separators

    def log_n_junction_trees(self, seps):
        lm = 0.0
        for sep in seps:
            lm += self.log_nu(sep)
        return lm

    def tuple(self):
        return (frozenset(self.nodes()), frozenset([frozenset(e) for e in self.edges()]))

    def __hash__(self):
        return hash(self.tuple())


def is_junction_tree(tree):
    """ Checks the junction tree property of a graph.

    Args:
        tree (NetworkX graph): A junction tree

    Returns:
        bool: True if tree is a junction tree
    """
    for n1 in tree.nodes():
        for n2 in tree.nodes():
            if n1 == n2:
                continue
            if n1 <= n2:
                return False

    for n1 in tree.nodes():
        for n2 in tree.nodes():
            if n1 == n2:
                continue
            inter = n1 & n2
            path = nx.shortest_path(tree, source=n1, target=n2)
            for n in path:
                if not inter <= n:
                    return False
    return True


def n_junction_trees(p):
    """ Returns the number of junction trees with p internal nodes.

    Args:
        p (int): number of internal nodes
    """
    graphs = glib.all_dec_graphs(p)
    num = 0
    for g in graphs:
        seps = glib.separators(g)
        jt = glib.junction_tree(g)
        num += int(np.exp(log_n_junction_trees(jt, seps)))
    return num


def induced_subtree(tree, s):
    """ Returns the subtree induced by the set s.

    Args:
       tree (NetworkX graph): A junction tree.
       s (set): Subset of the node in the underlying graph of T.
    """
    if len(s) == 0:
        return tree.subgraph(tree.nodes())
    v_prime = {c for c in tree.nodes() if s <= c}
    return tree.subgraph(v_prime)


def induced_subtree_nodes(tree, node, visited, sep):
    neigs = [n for n in tree.neighbors(node)
             if sep <= node and n not in visited]
    visited.add(node)
    if len(neigs) > 0:
        neig = neigs.pop()
        for neig in neigs:
            induced_subtree_nodes(tree, neig, visited, sep)
    return visited


def forest_induced_by_sep(tree, s):
    """ Returns the forest created from the subtree induced by s
    and cut at the separator that equals s.
    This is the forest named F in

    Args:
        tree (NetworkX graph): A junction tree
        s (set): A separator of tree

    Returns:
        NetworkX graph: The forest created from the subtree induced by s
    and cut at the separator that equals s.
    """
    F = induced_subtree(tree, s)
    edges_to_remove = []
    for e in F.edges():
        if s == e[0] & e[1]:
            edges_to_remove.append(e)
    F.remove_edges_from(edges_to_remove)
    return F


def separators(tree):
    """ Returns a dictionary of separators and corresponding
    edges in the junction tree tree.

    Args:
        tree (NetworkX graph): A junction tree

    Returns:
        dict:  Example {sep1: [sep1_edge1, sep1_edge2, ...], sep2: [...]}
    """
    separators = {}
    for edge in tree.edges():
        sep = edge[0] & edge[1]
        if not sep in separators:
            separators[sep] = set([])
        separators[sep].add(edge)
    return separators


def log_nu(tree, s):
    """ Returns the number of equivalent junction trees for tree where tree is cut at the separator s and then constructed again.

    Args:
        tree (NetworkX graph): A junction tree
        s (set): A separator of tree

    Returns:
        float
    """
    f = np.array(n_subtrees(tree, s))
    ts = f.ravel().sum()
    ms = len(f) - 1
    return np.log(f).sum() + np.log(ts) * (ms - 1)


def n_subtrees_aux(tree, node, sep, visited, start_nodes):
    visited.add(node)
    for n in nx.neighbors(tree, node):
        if sep < n:
            if n not in visited:
                if n & node == sep:
                    start_nodes.add(n)
                else:
                    n_subtrees_aux(tree, n, sep, visited, start_nodes)


def n_subtrees(tree, sep):
    if tree.size() == 0:
        return [1]
    visited = set()
    start_nodes = set()
    leaf = None
    counts = []
    for n in tree.nodes():
        valid_neighs = [ne for ne in nx.neighbors(tree, n) if sep < ne]
        if len(valid_neighs) == 1 and sep < n:
            leaf = n
            break

    start_nodes.add(leaf)
    prev_visited = 0
    while len(start_nodes) > 0:
        n = start_nodes.pop()
        n_subtrees_aux(tree, n, sep, visited, start_nodes)
        counts += [len(visited) - prev_visited]
        prev_visited = len(visited)

    return counts

def log_n_junction_trees(tree, S):
    """ Returns the number of junction trees equivalent to tree where trees
    is cut as the separators in S. is S i the full set of separators in tree,
    this is the number of junction trees equivalent to tree.

    Args:
        tree (NetworkX graph): A junction tree
        S (list): List of separators of tree

    Returns:
        float
    """
    log_mu = 0.0
    for s in S:
        log_mu += log_nu(tree, s)
    return log_mu


def randomize_at_sep(tree, s):
    """ Returns a junction tree equivalent to tree where tree is cut at s
    and then reconstructed at random.

    Args:
        tree (NetworkX graph): A junction tree
        s (set): A separator of tree

    Returns:
        NetworkX graph
    """
    F = forest_induced_by_sep(tree, s) # should be junction tree
    new_edges = random_tree_from_forest(F)
    # Remove old edges associated with s
    to_remove = []
    for e in tree.edges():  # TODO, get these easier
        if e[0] & e[1] == s:
            to_remove += [(e[0], e[1])]

    tree.remove_edges_from(to_remove)
    # Add the new edges

    for e in new_edges:
        tree.add_edge(e[0], e[1])


def randomize(tree):
    """ Returns a random junction tree equivalent to tree.

    Args:
        tree (NetworkX graph): A junction tree
        s (set): A separator of tree

    Returns:
        NetworkX graph
    """
    S = separators(tree)
    for s in S:
        randomize_at_sep(tree, s)


def random_tree_from_forest(F, edge_label=""):
    """ Returns a random tree from a the forest F.

    Args:
        F (NetworkX graph): A forest.
        edge_label (string): Labels for the edges.
    """
    comps = [list(c) for c in nx.connected_components(F)]

    q = len(comps)
    p = F.order()
    # 1. Label the vertices's
    all_nodes = []
    for i, l in enumerate(comps):
        for j in range(len(l)):
            all_nodes.append((i, j))
    # 2. Construct a list v containing q - 2 vertices each chosen at
    #    random with replacement from the set of all p vertices.
    v_ind = np.random.choice(p, size=q-2)

    v = [all_nodes[i] for i in v_ind]
    v_dict = {}
    for (i, j) in v:
        if i not in v_dict:
            v_dict[i] = []
        v_dict[i].append(j)

    # 3. Construct a set w containing q vertices,
    # one chosen at random from each subtree.
    w = []
    for i, c in enumerate(comps):
        # j = np.random.choice(len(c))
        j = np.random.randint(len(c))
        w.append((i, j))

    # 4. Find in w the vertex x with the largest first index that does
    #    not appear as a first index of any vertex in v.
    edges_ind = []
    while not v == []:
        x = None
        #  not in v
        for (i, j) in reversed(w):  # these are ordered
            if i not in v_dict:
                x = (i, j)
                break

        # 5. and 6.
        y = v.pop()  # removes from v
        edges_ind += [(x, y)]
        del v_dict[y[0]][v_dict[y[0]].index(y[1])]  # remove from v_dict
        if v_dict[y[0]] == []:
            v_dict.pop(y[0])
        del w[w.index(x)]  # remove from w_dict

    # 7.
    edges_ind += [(w[0], w[1])]
    edges = [(comps[e[0][0]][e[0][1]], comps[e[1][0]][e[1][1]])
             for e in edges_ind]

    F.add_edges_from(edges, label=edge_label)
    return edges


def graph(tree):
    """ Returns the graph underlying the junction tree tree.

    Args:
        tree (NetworkX graph): A junction tree

    Returns:
        NetworkX graph
    """
    G = nx.Graph()
    for c in tree.nodes():
        for n1 in set(c):
            if len(c) == 1:
                G.add_node(n1)
            for n2 in set(c) - set([n1]):
                G.add_edge(n1, n2)
    return G


def peo(tree):
    """ Returns a perfect elimination order and corresponding cliques, separators, histories, , rests for tree.

    Args:
        tree (NetworkX graph): A junction tree.

    Returns:
       tuple: A tuple of form (C, S, H, A, R), where the elemenst are lists of Cliques, Separators, Histories, , Rests, from a perfect elimination order.
    """
    C = list(nx.dfs_preorder_nodes(tree, tree.nodes()[0]))
    S = [set() for j in range(len(C))]
    H = [set() for j in range(len(C))]
    R = [set() for j in range(len(C))]
    A = [set() for j in range(len(C)-1)]
    S[0] = None
    H[0] = C[0]
    R[0] = C[0]
    for j in range(1, len(C)):
        H[j] = H[j-1] | C[j]
        S[j] = H[j-1] & C[j]
        A[j-1] = H[j-1] - S[j]
        R[j] = C[j] - H[j-1]
    return (C, S, H, A, R)