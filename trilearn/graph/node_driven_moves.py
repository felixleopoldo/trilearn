
import networkx as nx
import numpy as np
import random
import scipy.special as sp
from trilearn.graph import junction_tree as jtlib


def leaf_nodes(tree, neiedge=False):
    """ Returns the leafe node of tree
    Args:
      tree (NetworkX graph): a tree
      neiedge (logical): True for returning the nei cliques as well, False just
      the clique
    Return:
        a set of tupples (leaf clique, neighboring to leaf clique)
        when neiedge = True, otherwise a set of leaves
    """
    return {x for x in tree.nodes() if tree.degree(x) == 1}


def graph_node_driven_moves(tree, node, dumble=False, add_empty_node=True):
    """ Return boundary cliques for a specific node
    Args:
      tree (NetwokrX) a junction tree
      node (integer) a node
    """
    boundary_cliques = set()
    nei_cliques = dict()
    if len([node]) == 0:
        return None
    else:
        if not type(node) is frozenset:
            node = frozenset([node])

    # boundary cliques (disconnect move)
    T = jtlib.subtree_induced_by_subset(tree, node)
    if len(T) == 1:             # num of nodes is 1
        None
    elif len(T) == 2:              # a dumble (2 nodes) return a random one
        if dumble:
            boundary_cliques = {x for x in T.nodes()}
        else:
            r = np.random.randint(2)  # choose a random clique
            boundary_cliques = {frozenset(list(T.nodes())[r])}
    else:
        boundary_cliques = leaf_nodes(T)
    boundary_cliques = {x for x in boundary_cliques if x - node not in tree}
    # nei_cliques (connect move)
    for subnode in T:           # subnode is not necessary a boundary clique
        for nei in tree.neighbors(subnode):
            if (not node & nei) and ((node | nei) not in T):
                if subnode in nei_cliques.keys():
                    nei_cliques[subnode].append(nei)
                else:
                    nei_cliques[subnode] = [nei]
    # adding single-clique node
    if add_empty_node and len(tree) < tree.num_graph_nodes:
        if node not in T:
            r = np.random.choice(len(T))
            subnode = list(T.nodes())[r]
            if subnode in nei_cliques.keys():
                nei_cliques[subnode].append(frozenset())
            else:
                nei_cliques[subnode] = [frozenset()]
    return boundary_cliques, nei_cliques


def all_possible_moves(tree, node, dumble=False):
    """ Returns a set of boundary cliques that are not associated with 
        neiboring cliques, a dictionary of neighboring cliques that are not
        associated with boundary cliques (they keys of the dict are the
        connectors), and a dict of neiboring cliques that are connected
        to boundary cliques (key).
    Args:
      tree (NetwokrX) a junction tree
      node (integer) a node
    """
    bd_cliques, nei_cliques = graph_node_driven_moves(tree, node, False)

    nei_and_bd = dict()
    nei_no_bd = dict()
    bd_no_nei = set()
    for x in nei_cliques.keys():
        if x in bd_cliques:
            nei_and_bd[x] = nei_cliques[x]
        else:
            nei_no_bd[x] = nei_cliques[x]

    bd_no_nei = bd_cliques.difference(nei_and_bd.keys())

    return bd_no_nei, nei_no_bd, nei_and_bd


# def all_possible_moves(tree, node, dumble=False):
#     """ Returns a dictionary of buckets, of each bucket one connection is
#         possible at a time. Neighboring cliques are sets of two nodes,
#         and boundary cliques are sets of a single nodes.
#     Args:
#       tree (NetwokrX) a junction tree
#       node (integer) a node
#     """
#     if len([node]) == 0:
#         return None
#     else:
#         if not type(node) is frozenset:
#             node = frozenset([node])
#     bd_cliques, nei_cliques = graph_node_driven_moves(tree, node, dumble)

#     nei_bucket = dict()         # bucket for nei-with no bd cliques
#     boundary_bucket = dict()
#     for bd, nei in nei_cliques.items():
#         if not type(nei) is list:
#             nei = [nei]
#         if bd in bd_cliques:
#             # if bucket key exist
#             if bd in boundary_bucket.keys():
#                 boundary_bucket[bd].append(nei)
#             else:
#                 boundary_bucket[bd] = nei  # add the nei
#                 boundary_bucket[bd].append(bd)  # add the boundary clique
#         else:
#             for x in nei:
#                 nei_bucket[x] = [bd]

#     for x in bd_cliques.difference(boundary_bucket.keys()):
#         boundary_bucket[x] = [x]

#     return boundary_bucket, nei_bucket


def sample_1per_bucket(buckets):
    """ Return a set of nodes to move on, randomly selected buckets
    Args:
     buckets (dict) a dictionary of lists of nodes, where from each
        list one can select only one node
    """
    buckets_new = dict()
    m = [len(x) for x in buckets.values()]
    for key, item in buckets.items():
        n = len(item)
        r = np.random.randint(n) if n > 1 else 0
        buckets_new[key] = item[r]
    return buckets_new, m


def disconnect(tree, old_node, new_node):
    if new_node not in tree:
        if len(new_node) != 0:     # in case of an empty tree-node
            tree.add_node(new_node)
            edges_to_add = [(new_node, y) for y in tree.neighbors(old_node)
                            if y != new_node]
            tree.add_edges_from(edges_to_add)
            tree.remove_node(old_node)
        else:
            tree.remove_node(old_node)
    else:
        print('node in tree -- disconnect {}'.format(new_node))
        


def connect(tree, old_node, new_node, connector_node=None):
    if new_node not in tree:
        tree.add_node(new_node)
        # import pdb; pdb.set_trace()
        if old_node:  # not an empty clique-node
            edges_to_add = [(new_node, y) for y in tree.neighbors(old_node)
                            if y != new_node]
            tree.remove_node(old_node)
            tree.add_edges_from(edges_to_add)
        else:  # empty clique-node
            edges_to_add = [(new_node, connector_node)]
            tree.add_edges_from(edges_to_add)
    else:
        print('node in tree -- connect {}'.format(new_node))


def propose_moves(tree, node):
    """ Proposes a random set of new moves, given the current state of
        the junction tree. Returns the set of new junction tree nodes, 
        after performing the move directly on the tree, and the probability
        of those moves.
    Args:
      tree (NetwokrX) a junction tree
      node (integer) a node
    """
    if not type(node) is set and not type(node) is frozenset:
        node = frozenset([node])

    bd_no_nei, nei_no_bd, nei_and_bd = all_possible_moves(tree, node)
    new_bd = set()
    keys = nei_and_bd.keys()
    for x in keys:
        if np.random.uniform(size=1) <= 0.5:
            new_bd.add(x)
            nei_and_bd.pop(x, None)

    bd_bucket = bd_no_nei.union(new_bd)
    nei_no_bd.update(nei_and_bd)
    n_bd = len(bd_bucket)
    nei_value_len = [len(x) for x in nei_no_bd.values()]
    n_nei = int(np.sum(nei_value_len))
    N = int(n_bd + n_nei)
    k = np.random.randint(N) + 1
    subset = np.random.choice(N, k, replace=False).tolist()
    bd_n = [i for i in subset if i < n_bd] if n_bd else None
    nei_n = [i - n_bd for i in subset if i >= n_bd] if n_nei else None
    new_nodes = set()
    if bd_n:
        bb = list(bd_bucket)
        for i in bd_n:
            old_node = bb[i]
            X = old_node - node
            disconnect(tree, old_node, X)
            new_nodes.add(X)
    if nei_n:
        keys = nei_no_bd.keys()
        aux = list(range(len(keys)))
        index = np.repeat(aux, nei_value_len).tolist()
        a = [index[i] for i in nei_n]
        values, counts = np.unique(a, return_counts=True)
        for i in range(len(values)):
            conn = keys[values[i]]
            j = counts[i]
            nei = nei_no_bd[conn]
            np.random.shuffle(nei)
            for old_node in nei[:j]:
                X = node | old_node
                connect(tree, old_node, X, conn)
                new_nodes.add(X)

    return new_nodes, log_prob(N, k, len(nei_no_bd.keys()))

def log_prob(n, k, m=0):
    """ returns the log probability of choosing k out of n
    Args:
    n (integer)
    k (interger) <= n
    m (integer) number of samples from each sampled bucket in n (generaly 1)
    """
    return - np.log(sp.binom(n, k)) - m*np.log(2)       # np.sum(np.log(m))


def extract(d, keys):
    return dict((k, d[k]) for k in keys if k in d)


def inverse_proposal_prob(new_nodes, tree, node):
    if len([node]) == 0:
        return None
    else:
        if not type(node) is frozenset:
            node = frozenset([node])

    bd_, nei_ = all_possible_moves(tree, node, True)
    n = len(bd_.keys()) + len(nei_.keys())
    k = len(new_nodes)
    subp = [1]
    for nd in new_nodes:
        if nd & node:           # bd_bucket
            subp.append(len(bd_[nd]))
        else:                   # nei node, might be in bd_bucket
            if nd and tree.neighbors(nd) in bd_.keys():
                for x in tree.neighbors(nd):
                    if x in bd_.keys():
                        subp.append(len(bd_[x]))
    subp = [1]
    return log_prob(n, k, subp), n, k, subp


def revert_moves(nodes, tree, node):
    if not type(node) is frozenset:
        node = frozenset([node])

    for nd in nodes:
        if node & nd:           # disconnect
            X = nd - node
            disconnect(tree, nd, X)
        else:       # connect move
            if nd:  # not an empty node
                X = node | nd
                connect(tree, nd, X)
            else:               # empty node
                X = node | nd
                T = jtlib.subtree_induced_by_subset(tree, node)
                conn = list(T.nodes() - nodes)[0]
                connect(tree, nd, X, conn)
