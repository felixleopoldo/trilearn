import itertools
import json

import numpy as np
import scipy.stats as stats

import trilearn.graph.decomposable
import trilearn.graph.graph as libg
import trilearn.graph.junction_tree as libj
import trilearn.auxiliary_functions as aux
from trilearn.distributions import dirichlet


def ll_complete_set_ratio(comp, alpha, counts, data, levels, cache):
    """ The ratio of normalizing constants for a posterior Dirichlet
    distribution defined ofer a complete set (clique or separator).
    I(alpha + n) / I(alpha)
    Args:
        comp: Clique or separator.
        alpha: Pseudo counts for each cell.
    """
    if comp not in counts:
        counts[comp] = aux.get_marg_counts(data, list(comp))
    if comp not in cache:
        nodes = list(comp)
        c1 = dirichlet.log_norm_constant_multidim(counts[comp],
                                                  alpha,
                                                  levels[nodes])

        c2 = dirichlet.log_norm_constant_multidim({},
                                                  alpha,
                                                  levels[nodes])

        cache[comp] = c1 - c2
    return cache[comp]


def log_likelihood_partial(cliques, separators, no_levels, cell_alpha, counts, data, levels, cache):
    cliques_constants = 0.0
    tot_no_cells = np.prod([l for l in no_levels])

    for c in cliques:
        # Setting constant alpha here
        no_cells_outside = np.prod([l for i, l in
                                    enumerate(no_levels) if
                                    i not in c])
        alpha = cell_alpha * no_cells_outside / tot_no_cells
        cliques_constants += ll_complete_set_ratio(c, alpha, counts, data, levels, cache)

    seps_constants = 0.0
    for s in separators:
        if s == frozenset({}):
            continue
        nu = len(separators[s])
        # Setting alpha here
        no_cells_outside = np.prod([l for i, l in
                                    enumerate(no_levels) if
                                    i not in s])
        alpha = cell_alpha * no_cells_outside / tot_no_cells
        seps_constants += nu * ll_complete_set_ratio(s, alpha, counts, data, levels, cache)

    return cliques_constants - seps_constants


def sample_hyper_consistent_counts(graph, levels, constant_alpha):
    """
    TODO
    """
    junctiontree = trilearn.graph.decomposable.junction_tree(graph)
    (C, S, H, A, R) = libj.peo(junctiontree)
    parameters = {}

    no_levels = np.array([len(l) for l in levels])

    for i, clique in enumerate(C):
        if i == 0:
            nodes = list(clique)
            no_cells = np.prod(no_levels[nodes])
            alphas = [constant_alpha/no_cells] * no_cells
            x = stats.dirichlet.rvs(alphas)
            x.shape = tuple(no_levels[nodes])
            parameters[clique] = x
        else:
            # Find clique that contains S[i]
            cont_clique = None
            for j in range(i):
                if S[i] <= C[j]:
                    cont_clique = C[j]
                    break
            (parameters[clique],
             parameters[S[i]]) = hyperconsistent_cliques(cont_clique,
                                                         parameters[cont_clique],
                                                         clique,
                                                         levels,
                                                         constant_alpha)

    return parameters


def sample_hyper_consistent_parameters(graph, constant_alpha, levels):
    junctiontree = trilearn.graph.decomposable.junction_tree(graph)
    (C, S, H, A, R) = libj.peo(junctiontree)
    parameters = {}

    no_levels = np.array([len(l) for l in levels])

    for i, clique in enumerate(C):
        if i == 0:
            nodes = list(clique)
            no_cells = np.prod(no_levels[nodes])
            alphas = [constant_alpha/no_cells] * no_cells
            x = stats.dirichlet.rvs(alphas)
            x.shape = tuple(no_levels[nodes])
            parameters[clique] = x
        else:
            # Find clique that contains S[i]
            cont_clique = None
            for j in range(i):
                if S[i] <= C[j]:
                    cont_clique = C[j]
                    break
            (parameters[clique],
            parameters[S[i]]) = hyperconsistent_cliques(cont_clique,
                                                        parameters[cont_clique],
                                                        clique,
                                                        levels,
                                                        constant_alpha)

    return parameters


def hyperconsistent_cliques(clique1, clique1_dist, clique2,
                            levels, constant_alpha):
    """ Returns a distribution for clique2 that is hyper-consistent
    with clique1_dist.

    Args:
        clique1 (set): A clique
        clique1_dist (np.array): A distribution for clique1
        clique2 (set): A clique
        levels (np.array of lists): levels for all nodes in the full graph
    """
    sep = list(clique1 & clique2)
    no_levels = np.array([len(l) for l in levels])
    clique2_dist_shape = tuple(no_levels[list(clique2)])

    if sep == set([]):
        alphas = [constant_alpha / np.prod(shape)] * np.prod(shape)
        clique2_dist = np.random.dirichlet(alphas).reshape()



    sep_dist_shape = tuple(no_levels[sep])
    clique2_dist = np.zeros(np.prod(no_levels[tuple(clique2)]),
                            dtype=np.float_).reshape(clique2_dist_shape)

    sep_dist = np.zeros(np.prod(no_levels[sep]),
                        dtype=np.float_).reshape(sep_dist_shape)

    for sepcells in itertools.product(*levels[sep]):
        # Set the indexing for clique2
        indexing_clique2 = [None]*len(clique2)
        for ind, node in enumerate(clique2):
            if node in sep:
                # Get index in separator set
                k = sep.index(node)
                indexing_clique2[ind] = sepcells[k]
            else:
                indexing_clique2[ind] = Ellipsis

        # Set the indexing for clique1
        indexing_clique1 = [None]*len(clique1)
        for ind, node in enumerate(clique1):
            print ind, node
            if node in sep:  # error if empty separator?
                # Get index in separator set
                k = sep.index(node)
                indexing_clique1[ind] = sepcells[k]
            else:
                indexing_clique1[ind] = Ellipsis

        print sep
        print clique1
        print clique1_dist
        print indexing_clique1
        print clique2
        shape = clique2_dist[tuple(indexing_clique2)].shape
        print "Bug?"  # TODO: Fix this bug, whatever it is...
        sep_marg = np.sum(clique1_dist[tuple(list(indexing_clique1))])
        sep_dist[sepcells] = sep_marg
        alphas = [constant_alpha / np.prod(shape)] * np.prod(shape)
        # alphas = [constant_alpha] * np.prod(shape)  # TODO
        d = np.random.dirichlet(alphas).reshape(shape)
        clique2_dist[tuple(indexing_clique2)] = d * sep_marg
    return (clique2_dist, sep_dist)
    # return clique2_dist


def prob_dec(x, parameters, cliques, separators):
    """ Probability of numpy array x in a decomposable model.
    """
    log_prob = 0.0
    for c in cliques:
        index = tuple(x[tuple(c)])
        log_prob += np.log(parameters[c].item(index))

    for s in separators:
        if len(s) > 0:
            index = tuple(x[tuple(s)])
            log_prob -= np.log(parameters[s].item(index))

    return np.exp(log_prob)


def conditional_prob_dec(x, y, dist, cliques, separators):
    """ Conditional probability of x given y, p(x | y).

    Args:
        x (dict):
    """
    prob = 1.0
    active_cliques = []
    active_separators = []

    for i, clique in enumerate(cliques):
        for node in x:
            if node in clique:
                active_cliques += [i]

        for node in y:
            if node in separators:
                active_separators += [i]

    return prob


def get_all_counts(graph, data):
    counts = {}
    junctiontree = trilearn.graph.decomposable.junction_tree(graph)
    for clique in junctiontree.nodes():
        counts[clique] = aux.get_marg_counts(data, list(clique))
    return counts


def est_parameters(graph, data, levels, const_alpha):
    counts = get_all_counts(graph, data)
    no_levels = np.array([len(l) for l in levels])
    n = float(len(data))
    parameters = {}
    for clique in counts:
        clique_levels = levels[list(clique)]
        clique_no_levels = no_levels[list(clique)]
        params = np.zeros(np.prod(clique_no_levels)).reshape(clique_no_levels)

        for cell in itertools.product(*clique_levels):
            params[cell] = const_alpha * \
                            np.prod(no_levels) / np.prod(clique_no_levels)
            if cell in counts[clique]:
                params[cell] += counts[clique][cell]

        params /= n + np.prod(no_levels)*const_alpha
        parameters[clique] = params

    return parameters


#def joint_prob_table(dist, levels, cliques, separators):
def joint_prob_table(graph, parameters, levels):
    (cliques, separators, _, _, _) = trilearn.graph.decomposable.peo(graph)

    if len(separators) > 1:
        separators = separators[1:]  # separators are counted from the second index
#        table = joint_prob_table(parameters, levels, C, S[1:])
#    else:
#        table = joint_prob_table(parameters, levels, C, [])


    no_levels = [len(l) for l in levels]
    table = np.zeros(np.prod(no_levels)).reshape(tuple(no_levels))
    for cell in itertools.product(*levels):
        table[cell] = prob_dec(np.array(cell), parameters, cliques, separators)
    return table


def sample_joint_prob_table(graph, levels, total_counts):
    local_tables = sample_hyper_consistent_parameters(graph, total_counts,
                                                      levels)
    table = joint_prob_table(graph, local_tables, levels)
    return table


def sample(table, n=1):
    p = len(table.shape)
    data = []
    for _ in range(n):
        x = []
        for i in range(p):
            dist = [None] * table.shape[i]
            for level in range(table.shape[i]):
                index = x + [level] + [Ellipsis]
                dist[level] = np.sum(table[index])
            dist /= sum(dist)
            val_bin = np.random.multinomial(1, dist)
            val = list(val_bin).index(1)
            x += [val]
        data.append(x)
    return np.array(data).reshape(n, p)


def read_local_hyper_consistent_parameters_from_json_file(filename):
    with open(filename) as data_file:
        json_parameters = json.load(data_file)

    no_levels = np.array(json_parameters["no_levels"])
    levels = [range(l) for l in no_levels]
    parameters = {}
    for clique_string, props in json_parameters.iteritems():
        if clique_string == "no_levels":
            continue
        clique = frozenset(props["clique_nodes"])
        clique_no_levels = tuple(no_levels[props["clique_nodes"]])
        distr = np.array(props["parameters"]).reshape(clique_no_levels)
        parameters[frozenset(props["clique_nodes"])] = distr

    return parameters