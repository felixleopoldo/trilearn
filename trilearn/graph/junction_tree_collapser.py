import itertools

import numpy as np

from trilearn.graph import junction_tree as jtlib
from trilearn.graph import graph as glib


def sample(tree, node):
    """ Removes node from the underlying decomposable graph of tree.
        two cases:
        If node was isolated, any junction tree representation of g(tree)\{node} is
        randomized at the empty separator.
        Otherwise, the origin node of each clique containing node is chosen
        deterministically.

    Args:
        tree (NetworkX graph): a junction tree
        node (int): a node for the underlying graph of tree

    Returns:
        NetworkX graph: a junction tree
    """

    #  shrinked_tree = tree.subgraph(tree.nodes()) # nx < 2.x
    shrinked_tree = tree.copy()  # nx > 2.x
    # If isolated node in the decomposable graph
    if frozenset([node]) in shrinked_tree.nodes():
        # Connect neighbors
        # neighs = tree.neighbors(frozenset([node])) # nx < 2.x
        neighs = list(tree.neighbors(frozenset([node])))  # nx > 2.x
        for j in range(len(neighs) - 1):
            shrinked_tree.add_edge(neighs[j], neighs[j + 1])
        shrinked_tree.remove_node(frozenset([node]))
        separators = shrinked_tree.get_separators()

        if frozenset() in separators:
            jtlib.randomize_at_sep(shrinked_tree, set())
    else:
        origins = {}
        Ncp = possible_origins(tree, node)

        #for c, neigs in Ncp.iteritems():
        for c, neigs in list(Ncp.items()):
            # take origin depending on if it was connected to all
            # in a clique or not
            # this is used when replicating the structure
            origin_c = None
            if neigs == []:
                origin_c = c - {node}
            else:
                origin_c = neigs[np.random.randint(len(neigs))]
            origins[c] = origin_c
            # Add the new clique
            shrinked_tree.add_node(origin_c) #, label=tuple(origin_c), color="blue")
            # Add neighbors
            # this might add c as a neighbor, but it is removed afterwards
            for neig in tree.neighbors(c):
                if not neig == origins[c]:
                    #lab = str(tuple(origin_c & neig))
                    shrinked_tree.add_edge(origin_c, neig)#, label=lab)

        # Remove the old cliques
        shrinked_tree.remove_nodes_from(origins)
        # Replicate subtree structure
        V_prime = {c for c in tree.nodes() if node in c}
        J_prime = tree.subgraph(V_prime)
        for e in J_prime.edges():
            #lab = str(tuple(origins[e[0]] & origins[e[1]]))
            shrinked_tree.add_edge(origins[e[0]], origins[e[1]])#, label=lab)

    return shrinked_tree


def support_subtree_nodes(tree, node):
    Ncp = possible_origins(tree, node)
    cps = list(Ncp.keys())
    subtree_nodes = itertools.product(*list(Ncp.values()))

    return cps, subtree_nodes


def sample_new(tree, node):
    """ Removes node from the underlying decomposable graph of tree.
        two cases:
        If node was isolated, any junction tree representation of g(tree)\{node} is
        randomized at the empty separator.
        Otherwise, the origin node of each clique containing node is chosen
        deterministically.

    Args:
        tree (NetworkX graph): a junction tree
        node (int): a node for the underlying graph of tree

    Returns:
        NetworkX graph: a junction tree
    """

    #  shrinked_tree = tree.subgraph(tree.nodes()) # nx < 2.x
    shrinked_tree = tree.copy()  # nx > 2.x
    # If isolated node in the decomposable graph
    if frozenset([node]) in shrinked_tree.nodes():
        # iterate over all trees that could be obtained by forming a tree
        # by connecting the forest obtained by removing the separators
        # corresponding to the empty set.
        # Connect neighbors
        # neighs = tree.neighbors(frozenset([node])) # nx < 2.x
        neighs = list(tree.neighbors(frozenset([node])))  # nx > 2.x
        for j in range(len(neighs) - 1):
            shrinked_tree.add_edge(neighs[j], neighs[j + 1])
        shrinked_tree.remove_node(frozenset([node]))
        separators = shrinked_tree.get_separators()

        if frozenset() in separators:
            jtlib.randomize_at_sep(shrinked_tree, set())
    else:

        Ncp = possible_origins(tree, node)
        cp_list = list(Ncp.keys())  # [frozenset([1, 2]), ...
        c_list = list(Ncp.values())  # [[frozenset([1, 2]), frozenset([2, 3])], [..], ]

        subtree_nodes = itertools.product(*c_list)  # TODO: could be slow. Possible fix: replace sets by indices.
        random_neigh_inds = np.random.multinomial(1, np.ones(len(subtree_nodes)) / len(subtree_nodes))
        neig_inds = subtree_nodes[random_neigh_inds]

        for i in range(len(cp_list)):
            c = c_list[i][neig_inds[i]]
            glib.replace_node(shrinked_tree, cp_list[i], c)

    return shrinked_tree


def support(tree, node):
    """ Removes node from the underlying decomposable graph of tree.
        two cases:
        If node was isolated, any junction tree representation of g(tree)\{node} is
        randomized at the empty separator.
        Otherwise, the origin node of each clique containing node is chosen
        deterministically.

    Args:
        tree (NetworkX graph): a junction tree
        node (int): a node for the underlying graph of tree

    Returns:
        NetworkX graph: a junction tree
    """

    #  shrinked_tree = tree.subgraph(tree.nodes()) # nx < 2.x

    # If isolated node in the decomposable graph
    if frozenset([node]) in tree.nodes():
        # iterate over all trees that could be obtained by forming a tree
        # by connecting the forest obtained by removing the separators
        # corresponding to the empty set.
        None
    else:

        Ncp = possible_origins(tree, node)
        cp_list = list(Ncp.keys())  # [frozenset([1, 2]), ...
        c_list = list(Ncp.values())  # [[frozenset([1, 2]), frozenset([2, 3])], [..], ]

        subtree_nodes = itertools.product(*c_list)  # TODO: could be slow. Fix: replace sets by indices.

        support = []
        # For support
        for i in range(len(subtree_nodes)):
            shrinked_tree = tree.copy()  # nx > 2.x
            setting = subtree_nodes[i]
            for j in range(len(cp_list)):
                glib.replace_node(shrinked_tree, cp_list[i], subtree_nodes[i][j])
            support += [shrinked_tree]

    return support


def backward_jt_traj_sample(perms_traj, tree):
    """ Samples a backward trajectory of junction trees in the
    order defined by perms_traj.

    Args:
       perm_traj (list): list of m-combinations with up to p nodes,
                         where p is the number of nodes on the graph of tree.
       tree (NetworkX graph): a junction tree.

    Returns:
       list: list of junction trees containing nodes in perm_traj
    """
    p = len(perms_traj)
    jts = [None for i in range(p)]
    jts[p - 1] = tree
    n = p - 2
    while n >= 0:
        to_remove = list(set(perms_traj[n + 1]) - set(perms_traj[n]))[0]
        jts[n] = sample(jts[n + 1], to_remove)
        n -= 1
    return jts
#
#
# def possible_origins(tree, node):
#     """ For each clique in the subtree spanned by those containing node,
#     a list of neighbors from which the corresponding clique
#     could adhere from is returned.
#
#     Args:
#         tree (NetworkX graph): a junction tree
#         node (int): a node for the underlying graph of tree
#
#     Returns:
#         dict: dict of new cliques containing node and the cliques from which each new cliques could have emerged from
#     """
#     v_prime = {c for c in tree.nodes() if node in c}
#     D = {cp: cp - {node} for cp in v_prime}
#     # the neigbors that the cliques could come from
#     origins = {cp: None for cp in v_prime}
#     for cp in v_prime:
#         origins[cp] = [c for c in tree.neighbors(cp) if c & cp == D[cp]]
#
#     return origins


def possible_origins(tree, node):
    """ For each clique in the subtree spanned by those containing node,
    a list of neighbors from which the corresponding clique
    could adhere from is returned.

    Args:
        tree (NetworkX graph): a junction tree
        node (int): a node for the underlying graph of tree

    Returns:
        dict: dict of new cliques containing node and the cliques from which each new cliques could have emerged from
    """
    v_prime = {c for c in tree.nodes() if node in c}
    D = {cp: cp - {node} for cp in v_prime}
    # the neighbors that the cliques could come from
    origins = {cp: [] for cp in v_prime}
    for cp in v_prime:
        for c in tree.neighbors(cp):
            if c & cp == D[cp]:
                origins[cp] += [c]
        if origins[cp] == []:
            origins[cp] = [D[cp]]

    return origins


def possible_origins_and_sets(tree, node):
    """ For each clique in the subtree spanned by those containing node,
    a list of neighbors from which the corresponding clique
    could adhere from is returned.

    Args:
        tree (NetworkX graph): a junction tree
        node (int): a node for the underlying graph of tree

    Returns:
        dict: dict of new cliques containing node and the cliques from which each new cliques could have emerged from
    """
    v_prime = {c for c in tree.nodes() if node in c}
    D = {c: c - {node} for c in v_prime}
    # the neighbors that the cliques could come from
    origins = {cp: None for cp in v_prime}
    for cp in v_prime:
        # get separator set S[c]. Then m = c - S[c]
        origins[cp] = [{"c": c, "d": D[cp], "q": cp-D[cp], "m": c-S[c], "r": c - D[cp]}
                       for c in tree.neighbors(cp) if c & cp == D[cp]]

    return origins


def log_count_origins(tree, old_tree, node):
    """ The (log) number of possible junction trees with the internal
    node, node removed that tree could have been built from using the CTA.

    Args:
        tree (nx graph): junction tree
        old_tree (nx graph): junction tree, where node is removed from tree
        node (int): Node in underlying graph

    Returns:
        int: log of the number of possible junction trees with the internal node p removed that tree could have been built from using the CTA.
    """
    # Special case for isolated node
    if frozenset([node]) in tree.nodes():
        to_return = old_tree.log_nu(frozenset())
        return to_return

    # For the other cases
    Ncp = possible_origins(tree, node)
    return np.sum([np.log(max(len(Ncp[c]), 1)) for c in Ncp])


def log_pdf(tree, old_tree, node=None):
    """
    Args:
        tree (nx graph): junction tree
        old_tree (nx graph): junction tree, where node is removed from tree
        node (int): Node in underlying graph
    """
    return -log_count_origins(tree, old_tree, node)
