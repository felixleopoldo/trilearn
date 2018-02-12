"""
The Christmas tree algorithm and related functions.
The CTA expands a junction tree by a new node in
the underlying decomposable graph.
"""

import numpy as np
import networkx as nx

import chordal_learning.auxiliary_functions as aux
import chordal_learning.graph.junction_tree as jtlib
import chordal_learning.graph.graph as glib


def shrink(tree, node):
    """ Removes node from the underlying decomposable graph of tree.
        two cases:
        If node was isolated, any junction tree representation of g(tree)\{i} is
        choosen at random.
        Otherwise, the origin node of each clique containing node is choosed
        deterministicly.

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
       perm_traj (list): list of m-combinations with up to p nodes, where p is the number of nodes on the graph of tree.
       tree (NetworkX graph): a junction tree.

    Returns:
       list: list of junction trees containing nodes in perm_traj
    """
    p = len(perms_traj)
    jts = [None for i in range(p)]
    jts[p - 1] = tree
    #jts[p - 1].fix_graph()
    n = p - 2
    while n >= 0:
        to_remove = list(set(perms_traj[n + 1]) - set(perms_traj[n]))[0]
        jts[n] = shrink(jts[n + 1], to_remove)
        #jts[n].fix_graph() # TODO: Is this crucial?
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


def get_subtree_nodes(T1, T2, new):
    """ If the junction tree T1 is expanded to T2 by one internal node n, then the subtree choosed in T1
    is (almost) unique. Also, the subtree of T2 containing n is unique.
    This returns a dictionary of the cliques in the induced subtree of T2 as keys and the emerging cliques in T1 as values.

    Args:
        T1 (NetworkX graph): a junction tree
        T2 (NetworkX graph): a junction tree

    Returns:
        dict: a dictionary of the cliques in the induced subtree of T2 as keys and the emerging cliques in T1 as values.
    """
    # Get subtree of T2 induced by the new node
    T2_ind = jtlib.induced_subtree(T2, {new})
    T2_subtree_nodes = None
    # Find the subtree(2) in T1
    if T2_ind.has_node(frozenset([new])):
        # Isolated node.  Unique empty subtree
        T2_subtree_nodes = [{frozenset([new]): None}]

    elif T2_ind.order() == 1:
        # Look which is its neighbor
        c = T2_ind.nodes()[0]

        if T1.has_node(c - {new}):
            # if it was connected to everything in a clique
            T2_subtree_nodes = [{c: c - {new}}]
        else:
            # c always has at lest one neighbor and the separator c\new
            # for all of them.
            # We have to decide which of them was the emerging clique.
            # 3 cases:
            # 1) 1 neighbor: Trivial.
            # 2) 2 neighbors: Then it could be any of these.
            # 3) >2 neighbors: The emerging clique is the one that has the
            #                  others as a subset of its neighbors in T1
            neigs = T2.neighbors(c)
            possible_origins = [c1 for c1 in neigs if c1 & c == c - {new}]
            g = len(possible_origins)
            if g == 1:
                # This neighbor has to be the one it came from
                T2_subtree_nodes = [{c: possible_origins[0]}]
            elif g == 2 and len(neigs) == 2:
                # If there are 2 possible neighbors with the same separator
                T2_subtree_nodes = [{c: possible_origins[0]},
                                    {c: possible_origins[1]}]
            else:
                for neig in possible_origins:
                    if set(neigs) - {neig} <= set(T1.neighbors(neig)):
                        T2_subtree_nodes = [{c: neig}]
                        break
    else:
        tmp = {}
        # In this case the subtree nodes are uniquely defined by T1 and T2
        # Loop through all edges in T2 in order to extract the correct
        # subtree of T1. Note that, by construction we know that it has the same structure as
        # the induced subtree of T2.
        for e in T2_ind.edges():
            # Non-swallowed cliques, get all potential "emerging"
            # (cliques from where the new cliques could have have emerged) cliques.
            Ncp1 = [c for c in T2.neighbors(e[0]) if
                    c & e[0] == e[0] - {new}]
            Ncp2 = [c for c in T2.neighbors(e[1]) if
                    c & e[1] == e[1] - {new}]

            # If the clique was swallowed in the new clique,
            # there will be no neighbors, so the clique itself
            # (except from the new node) is the unique emerging clique.
            if Ncp1 == []:
                Ncp1 = [e[0] - {new}]
            if Ncp2 == []:
                Ncp2 = [e[1] - {new}]

            # Replicate the structure in T2
            for neig1 in Ncp1:
                for neig2 in Ncp2:
                    if T1.has_edge(neig1, neig2): # Know that this edge is unique
                        tmp[e[0]] = neig1
                        tmp[e[1]] = neig2
                        #ctmp = e[0] - {new} # Get z_j U q_j
                        #if ctmp & neig1 == ctmp: # How could this ever be false?
                        #    tmp[e[0]] = neig1
                        #    tmp[e[1]] = neig2
                        #else:
                        #    print "False"
                        #    tmp[e[1]] = neig1
                        #    tmp[e[0]] = neig2
            T2_subtree_nodes = [tmp] # TODO: Bug? Too indented?
    return T2_subtree_nodes


def sample(order, alpha=0.5, beta=0.5):
    """ Generates a junction tree with order internal nodes with the junction tree expander.

    Args:
        order (int): number of nodes in the underlying graph
        alpha (float): parameter for the subtree kernel
        beta (float): parameter for the subtree kernel
        directory (string): path to

    Returns:
        NetworkX graph: a junction tree
    """
    G = nx.Graph()
    # G = jtlib.JunctionTree()
    G.add_node(order[0], shape="circle")
    tree = glib.junction_tree(G)
    # print tree.nodes()
    # for n in tree.nodes():
    #     lab = tuple(n)
    #     if len(n) == 1:
    #         lab = "("+str(list(n)[0])+")"
    #     tree.node[n] = {"color": "black", "label": lab}

    for j in order[1:]:
        (tree, _, _, _, _, _) = expand(tree, j, alpha, beta)

    return tree


def sample_graph(order, alpha=0.5, beta=0.5):
    if type(order) is int:
        tree = sample(range(order), alpha, beta)
        return jtlib.graph(tree)
    elif type(order) is list:
        tree = sample(order, alpha, beta)
        return jtlib.graph(tree)


def expand(tree, i, alpha, beta, directory=None):
    """ Expands the junciton tree tree with the internal node i

    Args:
        tree (NetworkX graph): a junction tree
        i (int): new node to be added to the underlying graph of tree
        alpha (float): parameter for the subtree kernel
        beta (float): parameter for the subtree kernel
        directory (string): path to

    Returns:
        NetworkX graph: a junction tree

    """
    # for n in tree.nodes():
    #     lab = tuple(n)
    #     if len(n) == 1:
    #         lab = "(" + str(list(n)[0]) + ")"
    #     tree.node[n] = {"color": "black", "label": lab}
    # print tree.nodes()
    tree_new = tree.subgraph(tree.nodes())
    #old_G = jtlib.get_graph(tree)
    #(subtree, old_separators, probtree) = glib.random_subtree(tree, alpha, beta)

    # plotGraph(subtree, directory+"subtree_"+str(i)+".eps")
    # for n in subtree.nodes():
    #     tree_old.node[n] = {"color": "blue", "label": tuple(n)}
    #     if n in tree.nodes():
    #         tree.node[n] = {"color": "blue", "label": tuple(n)}

    # plotGraph(tree_old.subgraph(tree_old.nodes()),
    #           directory + "tree(" + str(i-1) + ")p.eps")

    (_, subtree_nodes, subtree_edges, subtree_adjlist,
     old_separators, prob_subtree) = glib.random_subtree(tree, alpha, beta, i)
    (old_cliques,
     new_cliques,
     new_separators,
     P,
     neig) = random_christmas_tree(i, tree_new, subtree_nodes, subtree_edges, subtree_adjlist)

    #conn_nodes = set()
    #for clique in new_cliques:
    #    conn_nodes |= clique

    # for n in tree.nodes():
    #     lab = tuple(n)
    #     if len(n) == 1:
    #         lab = "("+str(list(n)[0])+")"
    #     if n in new_cliques:
    #         tree.node[n] = {"color": "red", "label": lab}
    # plotGraph(tree.subgraph(tree.nodes()), directory+"tree("+str(i)+").eps")

    #G = jtlib.get_graph(tree)
    # G.node[i] = {"color": "red"}
    # for n in old_G:
    #     if n in conn_nodes:
    #         old_G.node[n] = {"color": "blue"}
    #         G.node[n] = {"color": "blue"}

    # plotGraph(G, directory+"G"+str(i)+".eps")
    # plotGraph(old_G, directory+"G"+str(i-1)+"p.eps")

    # Proposal kernel
    K_st = None
    if len(subtree_nodes) == 1:
        # There might be two possible subtrees so
        # we calculate the probabilities for these explicitly
        K_st = K_star(tree, tree_new, alpha, beta, i)
    else:
        K_st = prob_subtree
        for c in P:
            K_st *= P[c] * neig[c]
    return tree_new, K_st, old_cliques, old_separators, new_cliques, new_separators


def prob_christmas_tree(tree1, tree2, tree2_subtree_nodes, new):
    """ Returns the probability of generating tree2 from tree1 where the subtree is defined by tree2_subtree_nodes.

    Args:
        tree1 (NetworkX graph): A junction tree
        tree2 (NetworkX graph): A junction tree expanded from tree1
        tree2_subtree_nodes: Contains the cliques in the subgraph of tree2 with the corresponding cliques in tree1.

    Returns:
        float: Probability of generating tree2 from tree1 where the subtree is defined by tree2_subtree_nodes.
    """

    # if new is isolated in the underlying graph
    if len(tree2_subtree_nodes) == 1 and tree2_subtree_nodes.values()[0] is None:
        sep = frozenset([])
        c = frozenset([new])
        lognu = jtlib.log_nu(tree2, sep)
        return ({c: np.power(np.exp(lognu), -1.0)}, {c: 1.0})

    # Get the subtree induced by the nodes
    tree1_subtree = tree1.subgraph([c_t1 for c_t2, c_t1 in
                                    tree2_subtree_nodes.iteritems()])

    # Get the separating sets
    S = {c: set() for c_t2, c in tree2_subtree_nodes.iteritems()}
    for c_tree2, c in tree2_subtree_nodes.iteritems():
        for neig in tree1_subtree.neighbors(c):
            S[c] = S[c] | (c & neig)

    # Get the chosen internal nodes
    M = {}
    for c_tree2, c in tree2_subtree_nodes.iteritems():
        M[c] = c_tree2 - {new} - S[c]

    P = {}
    N = {}
    for c_tree2, c in tree2_subtree_nodes.iteritems():
        neigs = {neig for neig in tree1.neighbors(c) if
                 neig & c <= c_tree2 and neig not in tree1_subtree.nodes()}
        RM = c - S[c]
        gamma = tree1_subtree.order()
        sepCondition = len({neig for neig in nx.neighbors(tree1_subtree, c) if
                            S[c] == neig & c}) > 0 or gamma == 1
        N[c] = 1.0
        if sepCondition is False:
            # Every internal node in c belongs to a separator
            P[c] = np.power(2.0, - len(RM))
            if not len(c) + 1 == len(c_tree2):
                N[c] = np.power(2.0, -len(neigs))
        else:
            P[c] = 1.0
            if len(RM) > 1:
                P[c] = (1.0 / len(RM)) * np.power(2.0, -(len(RM) - 1.0)) * len(M[c])
                if not len(c) + 1 == len(c_tree2):
                    N[c] = np.power(2.0, -len(neigs))
    return (P, N)


def random_christmas_tree(new, tree, subtree_nodes, subtree_edges, subtree_adjlist):
    """ Returns a random CT from tree given subtree.

    Args:
        n (int): Node to be added to the graph of tree.
        tree (NetworkX graph): A junction tree.
        subtree (NetworkX graph): A subtree of tree.

    Returns:
        (NetworkX graph): a junction tree expanded from tree where n has been added to subtree.
    """
    new_separators = {}
    new_cliques = set()
    old_cliques = set()
    subtree_order = len(subtree_nodes)

    if subtree_order == 0:
        # If the tree, tree is empty (n isolated node),
        # add random neighbor.
        c = frozenset([new])
        new_cliques.add(c)
        c2 = tree.nodes()[0]
        tree.add_node(c, label=tuple([new]), color="red")
        tree.add_edge(c, c2, label=tuple([]))

        sep = frozenset()
        #tree.fix_graph()
        jtlib.randomize_at_sep(tree, sep)

        new_separators[sep] = [(c, c2)]
        # tree TODO: the actual value for the key is not needed.
        P = {c: 1.0 / np.exp(tree.log_nu(sep))}
        return (old_cliques, new_cliques, new_separators, P, {c: 1.0})

    S = {c: set() for c in subtree_nodes}
    M = {c: set() for c in subtree_nodes}
    for c in S:
        for neig in subtree_adjlist[c]:
            S[c] = S[c] | (c & neig)
    RM = {c: c - S[c] for c in S}
    C = {c: set() for c in subtree_nodes}
    P = {}
    N_S = {c: set() for c in subtree_nodes}

    for c in RM:
        sepCondition = len({neig for neig in subtree_adjlist[c] if
                           S[c] == neig & c}) > 0 or len(subtree_adjlist) == 1

        if sepCondition is True:
            tmp = np.array(list(RM[c]))
            first_node = []
            if len(tmp) > 0:
                # Connect to one node
                first_ind = np.random.randint(len(tmp))
                first_node = tmp[[first_ind]]
                tmp = np.delete(tmp, first_ind)

            rest = set()
            if len(tmp) > 0:
                # Connect to the rest of the nodes if there are any left
                rest = aux.random_subset(tmp)
            M[c] = frozenset(rest | set(first_node))
        else:
            M[c] = frozenset(aux.random_subset(RM[c]))

    # Create the new cliques
    for clique in M:
        C[clique] = frozenset(M[clique] | S[clique] | {new})
        new_cliques.add(C[clique])

    # Get the neighbor set of each c which can be moved to C[c]

    for clique in subtree_nodes:
        N_S[clique] = {neig for neig in tree.neighbors(clique)
                       if neig & clique <= C[clique] and neig not in set(subtree_nodes)}
    # Add the new cliques
    for c in subtree_nodes:
        tree.add_node(C[c], label=str(tuple(C[c])), color="red")

    # Construct and add the new edges between the new cliques,
    # replicating the subtree
    for e in subtree_edges:
        if not C[e[0]] & C[e[1]] in new_separators:
            new_separators[C[e[0]] & C[e[1]]] = []
        new_separators[C[e[0]] & C[e[1]]].append((C[e[0]], C[e[1]]))

        # lab = str(tuple(C[e[0]] & C[e[1]]))
        # if len(C[e[0]] & C[e[1]]) == 1:
        #     lab = "(" + str(list(C[e[0]] & C[e[1]])[0]) + ")"
        # tree.add_edge(C[e[0]], C[e[1]],
        #               label=lab)
        tree.add_edge(C[e[0]], C[e[1]])

    # Move the neighbors of a swallowed node to the swallowing node
    # Remove the swallowed node
    for c in subtree_nodes:
        if C[c] - {new} == c:
            # If connecting to all nodes in a clique
            for neig in tree.neighbors(c):
                if neig not in subtree_nodes:
                    lab = str(tuple(C[c] & neig))
                    if len(C[c] & neig) == 1:
                        lab = "(" + str(list(C[c] & neig)[0]) + ")"
                    tree.add_edge(C[c], neig, label=lab)
            tree.remove_node(c)
            old_cliques.add(c)
        else:  # If not connecting to every node in a clique
            if not C[c] & c in new_separators:
                new_separators[C[c] & c] = []
            new_separators[C[c] & c].append((C[c], c))
            # lab = str(tuple(C[c] & c))
            # if len(C[c] & c) == 1:
            #     lab = "(" + str(list(C[c] & c)[0]) + ")"
            # tree.add_edge(C[c], c, label=lab)
            #print "adding edge: " + str((C[c], c))
            tree.add_edge(C[c], c)
            # Pick random subset of neighbors intersecting with subset of S U M

            N = aux.random_subset(N_S[c])
            for neig in N:
                # lab = str(tuple(C[c] & neig))
                # if len(C[c] & neig) == 1:
                #     lab = "(" + str(list(C[c] & neig)[0]) + ")"
                # tree.add_edge(C[c], neig, label=lab)
                tree.add_edge(C[c], neig)
            tree.remove_edges_from([(c, neig) for neig in N])

    # Compute probabilities
    N = {}
    for c in subtree_nodes:
        sepCondition = len({neig for neig in subtree_adjlist[c] if
                            S[c] == neig & c}) > 0 or len(subtree_adjlist) == 1

        if sepCondition is False:
            # Every internal node in c belongs to a separator
            P[c] = np.power(2.0, - len(RM[c]))
            if not len(c) + 1 == len(C[c]):
                N[c] = np.power(2.0, - len(N_S[c]))
            else:
                N[c] = 1.0
        else:
            P[c] = 1.0
            N[c] = 1.0
            if len(RM[c]) > 1:
                P[c] = (1.0 / len(RM[c]))
                P[c] *= np.power(2.0, - (len(RM[c]) - 1.0)) * len(M[c])
                if not len(c) + 1 == len(C[c]): # c not swallowed by C[c]
                    N[c] = np.power(2.0, - len(N_S[c]))

    # Remove the edges in tree
    tree.remove_edges_from(subtree_edges)
    return (old_cliques, new_cliques, new_separators, P, N)


def K_star(tree1, tree2, alpha, beta, new):
    """ CT kernel probability K(tree1, tree2)

    Args:
        tree1 (NetworkX graph): A junction tree
        tree2 (NetworkX graph): A junction tree
        alpha (float): Parameter for the subtree kernel
        beta (float): Parameter for the subtree kernel

    Returns:
       float: probability of generatin tree2 from tree1
    """
    prob = 0.0
    tree2_tree1_subtree_nodes = get_subtree_nodes(tree1, tree2, new)
    for tree2_subtree_nodes in tree2_tree1_subtree_nodes:
        tree1_subtree = tree1.subgraph([c_t1 for c_t2, c_t1 in
                                        tree2_subtree_nodes.iteritems() if c_t1 is not None])

        tree1_subtree_prob = glib.prob_subtree(tree1_subtree, tree1,
                                               alpha, beta)
        (P, N) = prob_christmas_tree(tree1, tree2, tree2_subtree_nodes, new)
        christtree_prob = np.prod([P[c] * N[c] for c in P])
        prob += tree1_subtree_prob * christtree_prob
    return prob


def mu_update(new_separators, from_tree, to_tree, log_old_mu):
    """ Returns the new log mu where to_tree has been generated from from_tree2

    Args:
        from_tree (NetworkX graph): A junction tree
        to_tree (NetworkX graph): A junction tree
        new_separators (dict): The separators generated by the CTA.
        log_old_mu: Log of the number of junction trees of from_tree.

    """
    return mu_update_ratio(new_separators, from_tree, to_tree) + log_old_mu


def mu_update_ratio(new_separators, from_tree, to_tree):
    """ Returns the log of the ratio of number of junction trees of from_tree and to_tree.

    Args:
        from_tree (NetworkX graph): A junction tree
        to_tree (NetworkX graph): A junction tree
        new_separators (dict): The separators generated by the CTA.
        log_old_mu (float): Log of the number of junction trees of from_tree.

    Returns:
        float: log(mu(to_tree/from_tree))
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