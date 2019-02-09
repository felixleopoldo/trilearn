import networkx as nx
import numpy as np

import trilearn
import trilearn.graph.subtree_sampler as ss
from trilearn import auxiliary_functions as aux


def sample(tree, i, alpha=0.5, beta=0.5, only_tree=True):
    """ Expands the junciton tree tree with the internal node i

    Args:
        tree (NetworkX graph): a junction tree
        i (int): new node to be added to the underlying graph of tree
        alpha (float): parameter for the subtree kernel
        beta (float): parameter for the subtree kernel

    Returns:
        NetworkX graph: a junction tree

    """
    # for n in tree.nodes():
    #     lab = tuple(n)
    #     if len(n) == 1:
    #         lab = "(" + str(list(n)[0]) + ")"
    #     tree.node[n] = {"color": "black", "label": lab}
    # print tree.nodes()

    #tree_new = tree.subgraph(tree.nodes()) # nx < 2.0
    tree_new = tree.copy() # nx < 2.0


    #old_G = trilearn.graph.junction_tree.get_graph(tree)
    #(subtree, old_separators, probtree) = glib.random_subtree(tree, alpha, beta)

    # plotGraph(subtree, directory+"subtree_"+str(i)+".eps")
    # for n in subtree.nodes():
    #     tree_old.node[n] = {"color": "blue", "label": tuple(n)}
    #     if n in tree.nodes():
    #         tree.node[n] = {"color": "blue", "label": tuple(n)}

    # plotGraph(tree_old.subgraph(tree_old.nodes()),
    #           directory + "tree(" + str(i-1) + ")p.eps")

    (_, subtree_nodes, subtree_edges, subtree_adjlist,
    old_separators, prob_subtree) = ss.random_subtree(tree, alpha, beta, i)
    (old_cliques,
     new_cliques,
     new_separators,
     P,
     neig) = sample_cond_on_subtree_nodes(i, tree_new, subtree_nodes, subtree_edges, subtree_adjlist)

    if only_tree is True:
        return tree_new
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

    #G = trilearn.graph.junction_tree.get_graph(tree)
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
        K_st = pdf(tree, tree_new, alpha, beta, i)
    else:
        K_st = prob_subtree
        for c in P:
            K_st *= P[c] * neig[c]
    return tree_new, K_st, old_cliques, old_separators, new_cliques, new_separators


def subtree_cond_pdf(tree1, tree2, tree2_subtree_nodes, new):
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
        lognu = trilearn.graph.junction_tree.log_nu(tree2, sep)
        return ({c: np.power(np.exp(lognu), -1.0)}, {c: 1.0})

    # Get the subtree induced by the nodes
    tree1_subtree = tree1.subgraph([c_t1 for c_t2, c_t1 in
                                    tree2_subtree_nodes.iteritems()])


    # Get the separating sets
    # S = sepsets_in_subgraph(tree2_subtree_nodes, tree1_subtree)
    S = {c: set() for c_t2, c in tree2_subtree_nodes.iteritems()}
    for c_tree2, c in tree2_subtree_nodes.iteritems():
        for neig in tree1_subtree.neighbors(c):
            S[c] = S[c] | (c & neig)

    # P, N get_subset_probabilities(tree2_)
    # Get the chosen internal nodes
    M = {}
    for c_tree2, c in tree2_subtree_nodes.iteritems():
        M[c] = c_tree2 - {new} - S[c]

    # Calculate probabilities corresponding to each clique
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


def sample_cond_on_subtree_nodes(new, tree, subtree_nodes, subtree_edges, subtree_adjlist):
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
        #c2 = tree.nodes()[0] # nx 1.9
        c2 = list(tree.nodes)[0] # nx 2.1
        tree.add_node(c, label=tuple([new]), color="red")
        tree.add_edge(c, c2, label=tuple([]))

        sep = frozenset()
        #tree.fix_graph()
        trilearn.graph.junction_tree.randomize_at_sep(tree, sep)

        new_separators[sep] = [(c, c2)]
        # tree TODO: the actual value for the key is not needed.
        P = {c: 1.0 / np.exp(tree.log_nu(sep))}
        return (old_cliques, new_cliques, new_separators, P, {c: 1.0})

    S = {c: set() for c in subtree_nodes}
    M = {c: set() for c in subtree_nodes}
    for c in S:
        for neig in subtree_adjlist[c]:
            #S[c] = S[c] | (c & neig)
            S[c] |= (c & neig)
    RM = {c: c - S[c] for c in S}
    C = {c: set() for c in subtree_nodes}
    P = {}
    N_S = {c: set() for c in subtree_nodes}

    sepCondition = {}
    for c in RM:
        sepCondition[c] = len({neig for neig in subtree_adjlist[c] if
                           S[c] == neig & c}) > 0 or len(subtree_adjlist) == 1

        if sepCondition[c] is True:
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
                       if neig & clique <= C[clique] and neig not in subtree_nodes}

    # Add the new cliques
    #for c in subtree_nodes:
    #    tree.add_node(C[c], label=str(tuple(C[c])), color="red")
    tree.add_nodes_from([C[c] for c in subtree_nodes])

    # Construct and add the new edges between the new cliques,
    # replicating the subtree
    for e in subtree_edges:
        sep = C[e[0]] & C[e[1]]
        if not sep in new_separators:
            new_separators[sep] = []
        new_separators[sep].append((C[e[0]], C[e[1]]))

        tree.add_edge(C[e[0]], C[e[1]])

    # Move the neighbors of a swallowed node to the swallowing node
    # Remove the swallowed node
    for c in subtree_nodes:
        if C[c] - {new} == c:
            # If connecting to all nodes in a clique
            for neig in tree.neighbors(c):
                if neig not in subtree_nodes:
                    tree.add_edge(C[c], neig)#, label=lab)

            tree.remove_node(c)
            old_cliques.add(c)
        else:  # If not connecting to every node in a clique
            sep = C[c] & c
            if not sep in new_separators:
                new_separators[sep] = []
            new_separators[sep].append((C[c], c))

            #print "adding edge: " + str((C[c], c))
            tree.add_edge(C[c], c)
            # Pick random subset of neighbors intersecting with subset of S U M

            N = aux.random_subset(N_S[c])
            #for neig in N:
            #    tree.add_edge(C[c], neig)

            tree.add_edges_from([(C[c], neig) for neig in N])
            tree.remove_edges_from([(c, neig) for neig in N])

    # Compute probabilities
    N = {}
    for c in subtree_nodes:
        #sepCondition = len({neig for neig in subtree_adjlist[c] if
        #                    S[c] == neig & c}) > 0 or len(subtree_adjlist) == 1

        if sepCondition[c] is False:
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
                P[c] = 1.0 / len(RM[c])
                P[c] *= np.power(2.0, - (len(RM[c]) - 1.0)) * len(M[c])
                if not len(c) + 1 == len(C[c]): # c not swallowed by C[c]
                    N[c] = np.power(2.0, - len(N_S[c]))

    # Remove the edges in tree
    tree.remove_edges_from(subtree_edges)
    return (old_cliques, new_cliques, new_separators, P, N)


def pdf(tree1, tree2, alpha, beta, new):
    """ CT kernel probability K(tree1, tree2)

    Args:
        tree1 (NetworkX graph): A junction tree
        tree2 (NetworkX graph): A junction tree
        alpha (float): Parameter for the subtree kernel
        beta (float): Parameter for the subtree kernel

    Returns:
       float: probability of generating tree2 from tree1
    """

    # TODO: Refactor so it can be used in M-H.
    prob = 0.0
    tree2_tree1_subtree_nodes = get_subtree_nodes(tree1, tree2, new)
    for tree2_subtree_nodes in tree2_tree1_subtree_nodes:
        tree1_subtree = tree1.subgraph([c_t1 for c_t2, c_t1 in tree2_subtree_nodes.iteritems() if c_t1 is not None])
        tree1_subtree_prob = ss.pdf(tree1_subtree, tree1, alpha, beta)

        (P, N) = subtree_cond_pdf(tree1, tree2, tree2_subtree_nodes, new)
        christtree_prob = np.prod([P[c] * N[c] for c in P])
        prob += tree1_subtree_prob * christtree_prob
    return prob


def get_subtree_nodes(T1, T2, new):
    """ If the junction tree T1 is expanded to T2 by one internal node new,
    then the subtree chosen in T1 is (almost) unique. Also, the subtree
    of T2 containing new is unique.
    This returns a dictionary of the cliques in the induced
    subtree of T2 as keys and the emerging cliques in T1 as values.

    Args:
        T1 (NetworkX graph): a junction tree
        T2 (NetworkX graph): a junction tree
        new (int): the new node

    Returns:
        dict: a dictionary of the cliques in the induced subtree
        of T2 as keys and the emerging cliques in T1 as values.
    """
    # Get subtree of T2 induced by the new node
    T2_ind = trilearn.graph.junction_tree.subtree_induced_by_subset(T2, {new})
    T2_subtree_nodes = None
    # Find the subtree(2) in T1
    if T2_ind.has_node(frozenset([new])):
        # Isolated node.  Unique empty subtree
        T2_subtree_nodes = [{frozenset([new]): None}]

    elif T2_ind.order() == 1:
        # Look which is its neighbor
        #c = T2_ind.nodes()[0] # nx < 2.x
        c = list(T2_ind.nodes)[0] # nx > 2.x

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
            #neigs = T2.neighbors(c) # nx < 2.x
            neigs = list(T2.neighbors(c)) # nx > 2.x
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
                    if T1.has_edge(neig1, neig2):  # Know that this edge is unique
                        tmp[e[0]] = neig1
                        tmp[e[1]] = neig2
        T2_subtree_nodes = [tmp]
    return T2_subtree_nodes