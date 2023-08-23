"""
Functions related to junction trees.
"""

import networkx as nx
import numpy as np

import trilearn.graph.junction_tree_expander as jte


class JunctionTree(nx.Graph):

    def __init__(self, data=None, **attr):
        nx.Graph.__init__(self, data, **attr)
        self.log_nus = {}
        self.separators = None

    def log_nu(self, sep):
        if sep not in self.log_nus:
            self.log_nus[sep] = log_nu(self, sep)
        return self.log_nus[sep]

    def fresh_copy(self):
        """Return a fresh copy graph with the same data structure.

        A fresh copy has no nodes, edges or graph attributes. It is
        the same data structure as the current graph. This method is
        typically used to create an empty version of the graph.

        Notes
        -----
        If you subclass the base class you should overwrite this method
        to return your class of graph.
        """
        return JunctionTree()

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        self.separators = None
        self.log_nus = {}
        return super(JunctionTree, self).add_edge(u_of_edge, v_of_edge, **attr)

    def add_edges_from(self, ebunch_to_add, **attr):
        self.separators = None
        self.log_nus = {}
        return super(JunctionTree, self).add_edges_from(ebunch_to_add, **attr)

    def remove_edge(self, u, v):
        self.separators = None
        self.log_nus = {}
        return super(JunctionTree, self).remove_edge(u, v)

    def remove_node(self, n):
        self.separators = None
        self.log_nus = {}
        return super(JunctionTree, self).remove_node(n)

    def remove_edges_from(self, ebunch):
        self.separators = None
        self.log_nus = {}
        return super(JunctionTree, self).remove_edges_from(ebunch)

    def get_separators(self):
        if self.separators is None:
            self.separators = separators(self)
        return self.separators

    def connected_component_vertices(self):
        return [list(c) for c in nx.connected_components(self)]

    def connected_components(self):
        return nx.connected_components(self)

    def log_n_junction_trees(self, seps):
        """Log of the number of junction tree obtained by cutting at seps.

        Args:
            seps (list): List of separators

        Returns:
            float: Log number of junction trees obtained by cutting at seps.
        """
        lm = 0.0
        for sep in seps:
            lm += self.log_nu(sep)
        return lm

    def to_graph(self):
        """ Returns the graph underlying this junction tree.

        Returns:
            NetworkX graph: The underlying graph.

        Example:
            >>> np.random.seed(1)
            >>> t = jtlib.sample(5)
            >>> t.edges
            EdgeView([(frozenset([1, 2]), frozenset([4])), (frozenset([1, 2]), frozenset([0, 2])), (frozenset([0, 2]), frozenset([3]))])
            >>> t.nodes
            NodeView((frozenset([1, 2]), frozenset([4]), frozenset([0, 2]), frozenset([3])))
            >>> g = t.to_graph()
            >>> g.nodes
            NodeView((0, 1, 2, 3, 4))
            >>> g.edges
            EdgeView([(0, 2), (1, 2)])           
        """

        G = nx.Graph()
        for c in self.nodes():
            for n1 in set(c):
                if len(c) == 1:
                    G.add_node(n1)
                for n2 in set(c) - set([n1]):
                    G.add_edge(n1, n2)
        return G

    def tuple(self):
        return(frozenset(self.nodes()), frozenset([frozenset(e) for e in self.edges()]))

    def __hash__(self):
        """Returns the hash value of this junction tree.

        Returns:
            integer: A unique hash value.
        """
        return hash(self.tuple())


def is_junction_tree(tree):
    """ Checks the junction tree property of tree.

    Args:
        tree (NetworkX graph): A junction tree.

    Returns:
        bool: True if tree is a junction tree.
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
    import trilearn.graph.decomposable as dlib

    graphs = dlib.all_dec_graphs(p)
    num = 0
    for g in graphs:
        seps = dlib.separators(g)
        jt = dlib.junction_tree(g)
        num += int(round(np.exp(log_n_junction_trees(jt, seps))))
    return num


def subtree_induced_by_subset(tree, s):
    """ Returns the subtree of tree induced by the nodes containing the set s.

    Args:
       tree (NetworkX graph): A junction tree.
       s (set): Subset of the node in the underlying graph of T.

    Example:
        >>> t = jtlib.sample(5)  
        >>> t.nodes
        NodeView((frozenset([0, 4]), frozenset([3]), frozenset([1, 2, 4])))
        >>> t.edges
        EdgeView([(frozenset([0, 4]), frozenset([1, 2, 4])), (frozenset([3]), frozenset([1, 2, 4]))])
        >>> subt = jtlib.subtree_induced_by_subset(t, frozenset([1]))
        >>> subt.nodes
        NodeView((frozenset([1, 2, 4]),))
        >>> t.edges
        EdgeView([(frozenset([0, 4]), frozenset([1, 2, 4])), (frozenset([3]), frozenset([1, 2, 4]))])
    """
    if len(s) == 0:
        return tree.copy()
    v_prime = {c for c in tree.nodes() if s <= c}
    return tree.subgraph(v_prime).copy()

def induced_subtree_nodes(tree, node, visited, sep):
    neigs = [n for n in tree.neighbors(node)
             if sep <= node and n not in visited]
    visited.add(node)
    if len(neigs) > 0:
        neigs.pop()
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
    F = subtree_induced_by_subset(tree, s)
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
    """ Returns the number of equivalent junction trees for tree where
        tree is cut at the separator s and then constructed again.

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
    #for n in nx.neighbors(tree, node):
    for n in tree.neighbors(node):
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
        #valid_neighs = [ne for ne in nx.neighbors(tree, n) if sep < ne]
        valid_neighs = [ne for ne in tree.neighbors(n) if sep < ne]
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
    F = forest_induced_by_sep(tree, s)
    new_edges = random_tree_from_forest(F)
    # Remove old edges associated with s
    to_remove = []
    for e in tree.edges():  # TODO, get these easier
        if e[0] & e[1] == s:
            to_remove += [(e[0], e[1])]

    tree.remove_edges_from(to_remove)

    # Add the new edges
    tree.add_edges_from(new_edges)
    #for e in new_edges:
    #    tree.add_edge(e[0], e[1])


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
    """ Returns a random tree from the forest F.

    Args:
        F (NetworkX graph): A forest.
        edge_label (string): Labels for the edges.
    """
    comps = F.connected_component_vertices()

    #comps = [list(c) for c in nx.connected_components(F)]
    #comps = [list(t.nodes()) for t in F.connected_components(prune=False)]
    q = len(comps)
    p = F.order()
    # 1. Label the vertices's
    all_nodes = []
    for i, comp in enumerate(comps):
        for j in range(len(comp)):
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
    """ Returns the graph underlying the junction tree.

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
    # C = list(nx.dfs_preorder_nodes(tree, tree.nodes()[0])) # nx < 2.x
    C = list(nx.dfs_preorder_nodes(tree, list(tree.nodes)[0])) # nx > 2.x
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


def n_junction_trees_update(new_separators, from_tree, to_tree, log_old_mu):
    """ Returns the new log mu where to_tree has been generated from from_tree2

    Args:
        from_tree (NetworkX graph): A junction tree
        to_tree (NetworkX graph): A junction tree
        new_separators (dict): The separators generated by the CTA.
        log_old_mu: Log of the number of junction trees of from_tree.

    """
    return log_n_junction_trees_update_ratio(new_separators, from_tree, to_tree) + log_old_mu


def log_n_junction_trees_update_ratio(new_separators, from_tree, to_tree):
    """ Returns the log of the ratio of number of junction trees of from_tree and to_tree.

    Args:
        from_tree (NetworkX graph): A junction tree
        to_tree (NetworkX graph): A junction tree
        new_separators (dict): The separators generated by the CTA.
        log_old_mu (float): Log of the number of junction trees of from_tree.

    Returns:
        float: log(mu(to_tree)/mu(from_tree))
    """

    old_full_S = from_tree.get_separators()
    new_full_S = to_tree.get_separators()
    old_subseps = set()
    new_subseps = set()

    # subtract those that has to be "re-calculated"
    for new_s in new_separators:
        for s in old_full_S:
            # the spanning tree for s will be different in the new tree
            # so the old calculation is removed
            if s <= new_s:
                old_subseps.add(s)
    for new_s in new_separators:
        for s in new_full_S:
            if s <= new_s:
                new_subseps.add(s)

    new_partial_mu = to_tree.log_n_junction_trees(new_subseps)
    old_partial_mu = from_tree.log_n_junction_trees(old_subseps)

    return new_partial_mu - old_partial_mu


def sample(internal_nodes, alpha=0.5, beta=0.5, only_tree=False):
    """ Generates a junction tree with order internal nodes with the junction tree expander.

    Args:
        internal_nodes (int): number of nodes in the underlying graph
        alpha (float): parameter for the subtree kernel
        beta (float): parameter for the subtree kernel
        directory (string): path to

    Returns:
        NetworkX graph: a junction tree
    """
    import trilearn.graph.decomposable as dlib
    nodes = None
    if type(internal_nodes) is int:
        nodes = range(internal_nodes)
    else:
        nodes = internal_nodes

    tree = JunctionTree()

    #from trilearn.graph.junction_tree_gt import JunctionTreeGT
    #tree = JunctionTreeGT()

    tree.add_node(frozenset([nodes[0]]))
    # print tree.nodes()
    # for n in tree.nodes():
    #     lab = tuple(n)
    #     if len(n) == 1:
    #         lab = "("+str(list(n)[0])+")"
    #     tree.node[n] = {"color": "black", "label": lab}

    for j in nodes[1:]:
        if only_tree:
            jte.sample(tree, j, alpha, beta, only_tree=only_tree)
        else:
            (tree, _, _, _, _, _) = jte.sample(tree, j, alpha, beta, only_tree=only_tree)

        #print("vert dict: " + str(tree.gp.vert_dict))
        #print("nodes: " + str(list(tree.vp.nodes)))

    return tree


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