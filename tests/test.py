import itertools

import numpy as np
import networkx as nx

import trilearn.graph.decomposable
import trilearn.graph.junction_tree as jtlib
import trilearn.graph.junction_tree_expander as jtexp
import trilearn.graph.junction_tree as libj
import trilearn.graph.graph as glib
import trilearn.graph.junction_tree_expander
import trilearn.graph.junction_tree_collapser as jtcol
import trilearn.graph.subtree_sampler

np.set_printoptions(precision=1)
np.set_printoptions(suppress=True)

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

def expand(tree, node, alpha, beta, directory=None):
    nodes = tree.nodes()
    #old_tree = tree.subgraph(nodes) # nx < 2.x
    old_tree = tree.copy() # nx > 2.x
    (subtree, subtree_nodes, subtree_edges, subtree_adjlist,
     old_separators, prob_subtree) = trilearn.graph.subtree_sampler.random_subtree(old_tree, alpha, beta, (node, node))
    (old_cliques,
    new_cliques,
    new_separators,
    P,
    neig) = trilearn.graph.junction_tree_expander.sample_cond_on_subtree_nodes(node, tree, subtree_nodes, subtree_edges, subtree_adjlist)

    K_st = trilearn.graph.junction_tree_expander.pdf(old_tree, tree, alpha, beta, node)
    return K_st


def ttest_collapser_support():
    tree = jtlib.sample(5)
    supp = jtcol.support(tree, 4)
    tree2 = jtcol.sample(4)
    assert tree2 in supp


def test_logmu():
    """
    Check so that logmu is equal to the example in Thomas & Green 2009.
    """
    cliques = [frozenset([11, 12]), frozenset([9, 12, 17]), frozenset([3, 7, 17, 22]),
    frozenset([9, 10]), frozenset([6]), frozenset([4]), frozenset([8, 17]),
    frozenset([17, 21]), frozenset([3, 18, 19]), frozenset([2, 3, 18]),
    frozenset([2, 3, 16]), frozenset([3, 20]), frozenset([2, 3, 18]), frozenset([1, 2, 3]),
    frozenset([3, 5]), frozenset([13, 14, 15]), frozenset([13, 14, 23])]
    edges = [(frozenset([11, 12]), frozenset([9, 12, 17])),
    (frozenset([9, 12, 17]), frozenset([9, 10])),
    (frozenset([9, 12, 17]), frozenset([3, 7, 17, 22])),
    (frozenset([3, 7, 17, 22]), frozenset([6])),
    (frozenset([3, 7, 17, 22]), frozenset([8, 17])),
    (frozenset([3, 7, 17, 22]), frozenset([3, 18, 19])),
    (frozenset([6]), frozenset([4])),
    (frozenset([8, 17]), frozenset([17, 21])),
    (frozenset([3, 18, 19]), frozenset([2, 3, 18])),
    (frozenset([2, 3, 18]), frozenset([2, 3, 16])),
    (frozenset([2, 3, 18]), frozenset([3, 20])),
    (frozenset([2, 3, 18]), frozenset([1, 2, 3])),
    (frozenset([1, 2, 3]), frozenset([3, 5])),
    (frozenset([1, 2, 3]), frozenset([13, 14, 15])),
    (frozenset([13, 14, 15]), frozenset([13, 14, 23]))]

    g = nx.Graph()
    g.add_nodes_from(cliques)
    g.add_edges_from(edges)
    S = libj.separators(g)

    assert int(np.round(np.exp(libj.log_n_junction_trees(g, S)))) == 57802752

def test_transprob():
    p = 4
    trees = [set() for _ in range(p)]
    trans_probs = {}
    N = 5000

    alpha = 0.5
    beta = 0.5
    # Check that the transition probabilities sums to 1.
    for i in range(N):
        tree = jtlib.JunctionTree()
        tree.add_node(frozenset([0]))
        trees[0].add(tree.tuple())
        for n in range(1, p):
            tree_tuple_old = tree.tuple()
            K = expand(tree, n, alpha, beta)
            tree_tuple = tree.tuple()
            if not tree_tuple_old in trans_probs:
                # trans_probs[tree_tuple_old] = set()
                trans_probs[tree_tuple_old] = {}
            trans_probs[tree_tuple_old].update({tree_tuple: K})
            # trans_probs[tree_tuple_old].add((tree_tuple, K))
            trees[n].add(tree_tuple)

    for tree, exp_trees in trans_probs.items():
        # print exp_trees
        sum = 0.0
        for key, val in exp_trees.items():
            # print val
            sum += val
        assert(np.abs(sum - 1.0) < 0.0001)
    assert len(trans_probs) == 13
    assert [len(trees[i]) for i in range(p)] == [1, 2, 10, 108]


def test_logmu_monte_carlo():
    p = 4
    matspace = [range(2) for i in range(p * p)]
    graphs = {}
    graph_jtreps = {}
    for adjmatvec in itertools.product(*matspace):
        adjmat = np.matrix(adjmatvec).reshape(p, p)
        if np.diag(adjmat).sum() == 0:
            g = nx.from_numpy_matrix(adjmat)
            if nx.is_chordal(g):
                if check_symmetric(adjmat):
                    # print libg.mu(g)
                    graphs[adjmatvec] = trilearn.graph.decomposable.n_junction_trees(g)
                    jt = trilearn.graph.decomposable.junction_tree(g)
                    S = jt.get_separators()
                    graph_jtreps[adjmatvec] = {"graph": g,
                                               "jts": set(),
                                               "mu": np.exp(jtlib.log_n_junction_trees(jt, S))}
                    for i in range(100):
                        jtlib.randomize(jt)
                        graph_jtreps[adjmatvec]["jts"].add(jt.tuple())

    #print graphs
    #print "Exact number of chordal graphs: " + str(len(graph_jtreps))
    #print "Exact number of junction trees: " + str(np.sum([val for key, val in graphs.iteritems()]))
    sum = 0
    for graph, val in graph_jtreps.items():
        #print val["mu"], val["jts"]
        #print val["mu"] - len(val["jts"])
        sum += len(val["jts"])
        assert np.abs(val["mu"] - len(val["jts"]) < 0.0000001)

    print(sum)
