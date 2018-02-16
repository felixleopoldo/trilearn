import numpy as np

from trilearn.graph import junction_tree as jtlib


def shrink(tree, node):
    """ Removes node from the underlying decomposable graph of tree.
        two cases:
        If node was isolated, any junction tree representation of g(tree)\{i} is
        chosen at random.
        Otherwise, the origin node of each clique containing node is choosed
        deterministically.

    Args:
        tree (NetworkX graph): a junction tree
        node (int): a node for the underlying graph of tree

    Returns:
        NetworkX graph: a junction tree
    """

    shrinked_tree = tree.subgraph(tree.nodes())
    # If isolated node in the decomposable graph
    if frozenset([node]) in shrinked_tree.nodes():
        # Connect neighbors
        neighs = tree.neighbors(frozenset([node]))
        for j in range(len(neighs) - 1):
            shrinked_tree.add_edge(neighs[j], neighs[j + 1])
        shrinked_tree.remove_node(frozenset([node]))
        separators = shrinked_tree.get_separators()

        if frozenset() in separators:
            jtlib.randomize_at_sep(shrinked_tree, set())
    else:
        origins = {}
        Ncp = possible_origins(tree, node)

        for c, neigs in Ncp.iteritems():
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
            shrinked_tree.add_node(origin_c,
                                   label=tuple(origin_c),
                                   color="blue")
            # Add neighbors
            # this might add c as a neighbor, but it is removed afterwards
            for neig in tree.neighbors(c):
                if not neig == origins[c]:
                    lab = str(tuple(origin_c & neig))
                    shrinked_tree.add_edge(origin_c, neig, label=lab)

        # Remove the old cliques
        shrinked_tree.remove_nodes_from(origins)
        # Replicate subtree structure
        V_prime = {c for c in tree.nodes() if node in c}
        J_prime = tree.subgraph(V_prime)
        for e in J_prime.edges():
            lab = str(tuple(origins[e[0]] & origins[e[1]]))
            shrinked_tree.add_edge(origins[e[0]], origins[e[1]], label=lab)

    return shrinked_tree


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
        jts[n] = shrink(jts[n + 1], to_remove)
        n -= 1
    return jts


def possible_origins(tree, node):
    """ For each clique in the subtree spanned by those containing p,
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
        origins[cp] = [c for c in tree.neighbors(cp) if c & cp == D[cp]]

    return origins


def log_count_origins(tree, old_tree, node):
    """ This is the (log) number of possible junction trees with the internal
    node p removed that tree could have been built from using the CTA.

    Args:
        tree (nx graph): junction tree
        old_tree (nx graph): junction tree
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