
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
    if not type(node) is set and not type(node) is frozenset:
        node = frozenset([node])

    bd_bucket, nei_bucket = all_possible_moves(tree, node)
    if not bd_bucket and not nei_bucket:
        return set(), np.log(.0001)
    n_single = len(bd_bucket)
    m_ = len(nei_bucket.keys())
    n = n_single + m_
    sample_n = np.random.randint(n) + 1
    # sample_n = 1
    subset = np.random.choice(n, sample_n, replace=False).tolist()
    bd_n = [i for i in subset if i < n_single] if n_single else None
    nei_n = [i - n_single for i in subset if i >= n_single] if m_ else None
    new_nodes = set()
    subp = [1]
    if bd_n:
        single_bucket_moves, p = sample_1per_bucket(bd_bucket)
        keys = list(single_bucket_moves.keys())
        subkeys = [keys[i] for i in bd_n]
        subp = [p[i] for i in bd_n]
        sampled_bd_moves = extract(single_bucket_moves, subkeys)
        for connector_node, old_node in sampled_bd_moves.items():
            if node & old_node:         # disconnect move
                X = old_node - node
                disconnect(tree, old_node, X)
            else:       # connect move
                X = node | old_node
                connect(tree, old_node, X, connector_node)
            new_nodes.add(X)
    if nei_n:
        keys = list(nei_bucket.keys())
        subkeys = [keys[i] for i in nei_n]
        sampled_nei_moves = extract(nei_bucket, subkeys)
        for old_node, connector_node in sampled_nei_moves.items():
            if node & old_node:         # disconnect move
                X = old_node - node
                disconnect(tree, old_node, X)
            else:       # connect move
                X = node | old_node
                connect(tree, old_node, X, connector_node[0])
            new_nodes.add(X)
    subp = [1]
    return new_nodes, log_prob(n, sample_n, subp)


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
