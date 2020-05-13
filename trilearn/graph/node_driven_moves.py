
import networkx as nx
import numpy as np
import random
import scipy.special as sp
from trilearn.graph import junction_tree as jtlib


def leaf_nodes(tree):
    """ Returns a set of the leaf nodes of tree
    Args:
      tree (NetworkX graph): a tree
    """
    return {x for x in tree.nodes() if tree.degree(x) == 1}


def boundary_cliques_node(tree, node, *cache):
    """ Return boundary cliques for a specific node
    Args:
      tree (NetwokrX) a junction tree
      node (integer) a node
      cache (set) is used to pass cliques to check against, only used to calculate
                  the probability of the inverse move.
    """
    # boundary cliques (disconnect move)
    boundary_cliques = set()
    T = jtlib.subtree_induced_by_subset(tree, node)
    if len(T) == 1:             # num of nodes is 1
        None
    elif len(T) == 2:              # a prob (2 nodes) return a random one
        if cache:
            boundary_cliques = T.nodes() & cache[0]
        else:
            r = np.random.randint(2)  # choose a random clique
            boundary_cliques = {frozenset(list(T.nodes())[r])}
    else:
        boundary_cliques = leaf_nodes(T)
    boundary_cliques = {x for x in boundary_cliques if x - node not in tree}
    return boundary_cliques


def neighboring_cliques_node(tree, node, empty_node=True):
    """ Return neighboring cliques for the node-induced junction tree
    in  a dictionary. key:item pairs as (connector in node-induced):nei_clique
    Args:
      tree (NetwokrX) a junction tree
      node (integer) a node
      empty_node (bool) if empty cliques should be included
    """
    nei_cliques = dict()
    T = jtlib.subtree_induced_by_subset(tree, node)
    # nei_cliques (connect move)
    for subnode in T:           # subnode is not necessary a boundary clique
        for nei in tree.neighbors(subnode):
            if (not node & nei) and ((node | nei) not in T):
                if subnode in nei_cliques.keys():
                    nei_cliques[subnode].append(nei)
                else:
                    nei_cliques[subnode] = [nei]
    # adding single-clique node
    if empty_node and len(tree) < tree.num_graph_nodes:
        if node not in T:
            r = np.random.choice(len(T))
            subnode = list(T.nodes())[r]
            if subnode in nei_cliques.keys():
                nei_cliques[subnode].append(frozenset())
            else:
                nei_cliques[subnode] = [frozenset()]
    return nei_cliques


def propose_connect_moves(tree, node):
    """ Proposes a random set of connect moves, given the current state of
        the junction tree. Returns the set of new junction tree nodes,
        after performing the move directly on the tree, and the probability
        of those moves.
    Args:
      tree (NetwokrX) a junction tree
      node (integer) a node
    """
    if not type(node) is set and not type(node) is frozenset:
        node = frozenset([node])

    nei_cliques = neighboring_cliques_node(tree, node)
    if not nei_cliques:
        return [None] * 4
    nei_value_len = [len(x) for x in nei_cliques.values()]
    N = int(np.sum(nei_value_len))
    k = np.random.randint(N) + 1
    k = 1
    nei_n = np.random.choice(N, k, replace=False).tolist()
    new_cliques = set()
    if N > 0:
        keys = nei_cliques.keys()
        aux = list(range(len(keys)))
        index = np.repeat(aux, nei_value_len).tolist()
        a = [index[i] for i in nei_n]
        values, counts = np.unique(a, return_counts=True)
        for i in range(len(values)):
            conn = keys[values[i]]
            j = counts[i]
            nei = nei_cliques[conn]
            np.random.shuffle(nei)
            for old_node in nei[:j]:
                X = node | old_node
                connect(tree, old_node, X, conn)
                new_cliques.add(X)
    return new_cliques, log_prob(N, k, 1), N, k


def propose_disconnect_moves(tree, node, *cache):
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

    bd_cliques = boundary_cliques_node(tree, node, *cache)
    if not bd_cliques:
        return [None] * 4
    N = len(bd_cliques)
    k = np.random.randint(N) + 1
    k = 1
    subset = np.random.choice(N, k, replace=False).tolist()
    new_cliques = set()
    if N > 0:
        bb = list(bd_cliques)
        for i in subset:
            old_node = bb[i]
            X = old_node - node
            disconnect(tree, old_node, X)
            new_cliques.add(X)
    return new_cliques, log_prob(N, k, 1), N, k


def all_possible_moves(tree, node, empty_node=True, *cache):
    """ Returns a set of boundary cliques that are not associated with
        neiboring cliques, a dictionary of neighboring cliques that are not
        associated with boundary cliques (they keys of the dict are the
        connectors), and a dict of neiboring cliques that are connected
        to boundary cliques (key).
    Args:
      tree (NetwokrX) a junction tree
      node (integer) a node
    """
    if len([node]) == 0:
        return None
    else:
        if not type(node) is frozenset:
            node = frozenset([node])
   
    bd_cliques = boundary_cliques_node(tree, node, *cache)
    nei_cliques = neighboring_cliques_node(tree, node, empty_node)

    bd_nei = dict()
    nei_no_bd = dict()
    bd_no_nei = set()
    for x in nei_cliques.keys():
        if x in bd_cliques:
            bd_nei[x] = nei_cliques[x]
        else:
            nei_no_bd[x] = nei_cliques[x]

    bd_no_nei = bd_cliques.difference(bd_nei.keys())

    return bd_no_nei, nei_no_bd, bd_nei

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

    bd_no_nei, nei_no_bd, bd_nei = all_possible_moves(tree, node)
    new_bd = set()
    keys = bd_nei.keys()
    m = len(bd_nei.keys())
    for x in keys:
        if np.random.uniform(size=1) <= 0.5:
            new_bd.add(x)
            bd_nei.pop(x, None)

    bd_bucket = bd_no_nei.union(new_bd)
    nei_no_bd.update(bd_nei)
    n_bd = len(bd_bucket)
    nei_value_len = [len(x) for x in nei_no_bd.values()]
    n_nei = np.sum(nei_value_len)
    N = int(n_bd + n_nei)
    k = np.random.randint(N) + 1
    k = 1
    subset = np.random.choice(N, k, replace=False).tolist()
    bd_n = [i for i in subset if i < n_bd] if n_bd else None
    nei_n = [i - n_bd for i in subset if i >= n_bd] if n_nei else None
    new_cliques = set()
    updated_nodes = set()
    if bd_n:
        bb = list(bd_bucket)
        for i in bd_n:
            old_node = bb[i]
            X = old_node - node
            disconnect(tree, old_node, X)
            updated_nodes.add(old_node)
            new_cliques.add(X)
    bd_bucket = bd_bucket.difference(updated_nodes)
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
                new_cliques.add(X)
    return new_cliques, log_prob(N, k, 0), bd_bucket, nei_no_bd, N, m


def disconnect(tree, old_clique, new_clique):
    if new_clique not in tree:
        if len(new_clique) != 0:     # in case of an empty tree-node
            tree.add_node(new_clique)
            edges_to_add = [(new_clique, y) for y in tree.neighbors(old_clique)
                            if y != new_clique]
            tree.add_edges_from(edges_to_add)
            tree.remove_node(old_clique)
        else:
            if tree.degree(old_clique) != 1:
                # select a maximal clique
                for nei in tree.neighbors(old_clique):
                    if old_clique < nei:
                        edges_to_add = [(nei, y)
                                        for y in tree.neighbors(old_clique)
                                        if y != nei]
                        break
                tree.add_edges_from(edges_to_add)
            tree.remove_node(old_clique)
    else:
        print('node in tree -- disconnect {}'.format(new_clique))


def connect(tree, old_node, new_clique, connector_node=None):
    if new_clique not in tree:
        tree.add_node(new_clique)
        # import pdb; pdb.set_trace()
        if old_node:  # not an empty clique-node
            edges_to_add = [(new_clique, y) for y in tree.neighbors(old_node)
                            if y != new_clique]
            tree.remove_node(old_node)
            tree.add_edges_from(edges_to_add)
        else:  # empty clique-node
            edges_to_add = [(new_clique, connector_node)]
            tree.add_edges_from(edges_to_add)
    else:
        print('node in tree -- connect {}'.format(new_clique))


def log_prob(n, k, m=0):
    """ returns the log probability of choosing k out of n
    Args:
    n (integer)
    k (interger) <= n
    m (integer) nu
    """
    return - np.log(sp.binom(n, k)) - m*np.log(2)       # np.sum(np.log(m))


def inverse_proposal_prob(tree, node, new_cliques):
    """ Returns the log probability of the inverse propoal"""

    k = len(new_cliques)
    if node & new_cliques:         # inverse is disconnect
        bd_cliques = boundary_cliques_node(tree, node, new_cliques)
        N = len(bd_cliques)
    else:                       # inverse is connect
        nei_cliques = neighboring_cliques_node(tree, node, False)
        nei_value_len = [len(x) for x in nei_cliques.values()]
        N = int(np.sum(nei_value_len)) + 1*(len(tree) < tree.num_graph_nodes)
    return log_prob(N, k, 1), N, k


def revert_moves(tree, node, cliques):
    """ Revert moves in the junction tree
    Args:
      tree (NetwokrX) a junction tree
      node (integer) a node
      cliques (dict or set) of cliques to revert, diconnect if node in cliques
              otherwise connect.
    """
    # TODO: use type(cliques)=dict() to distinguish betwen cliques
    if not cliques:
        return None
    
    if not type(node) is frozenset:
        node = frozenset([node])

    for nd in cliques:
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
                conn = list(T.nodes() - cliques)[0]
                connect(tree, nd, X, conn)
