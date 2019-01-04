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
            nodes = sorted(list(clique))
            no_cells = np.prod(no_levels[nodes])
            alphas = [constant_alpha/no_cells] * no_cells
            x = stats.dirichlet.rvs(alphas)  # assume that the corresponding variables are ordered
            x.shape = tuple(no_levels[nodes])
            parameters[clique] = x
        else:
            # Find a clique that contains S[i]
            cont_clique = None
            for j in range(i):
                if S[i] < C[j]:
                    cont_clique = C[j]
                    break

            #print str(clique) + " neighbor of " + str(cont_clique)
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

    sep_list = sorted(list(clique1 & clique2))  # TODO: Bug, does not work if sorting this for some reason
    clique1_list = sorted(list(clique1))
    clique2_list = sorted(list(clique2))
    no_levels = np.array([len(l) for l in levels])
    clique2_dist_shape = tuple(no_levels[clique2_list])

    sep_dist_shape = tuple(no_levels[sep_list])
    clique2_dist = np.zeros(np.prod(no_levels[clique2_list]),
                            dtype=np.float_).reshape(clique2_dist_shape)

    sep_dist = np.zeros(np.prod(no_levels[sep_list]),
                        dtype=np.float_).reshape(sep_dist_shape)
    # we iterate through cell settings in the separators so we need
    # 1. to match the settings to clique settings.
    # 2. set Ellipsis the a node is not in a separator.

    for sepcells in itertools.product(*levels[sep_list]):
        # Set the indexing for the contingency/parameter table for clique2.
        # Since we iterate over the separators, we need to find it like this.
        indexing_clique2 = [None]*len(clique2) # Nodes are assumed to be in the same order as clique2
        for ind, node in enumerate(clique2_list):
            if node in sep_list:
                indexing_clique2[ind] = sepcells[sep_list.index(node)]
            else:
                indexing_clique2[ind] = slice(no_levels[node])

        # Set the indexing for clique1
        indexing_clique1 = [None]*len(clique1)
        for ind, node in enumerate(clique1_list):
            if node in sep_list:
                # Get index in separator set
                indexing_clique1[ind] = sepcells[sep_list.index(node)]
            else:
                indexing_clique1[ind] = slice(no_levels[node])

        # Calculate marginal distribution for the spe setting sepcells in clique1
        # this should then be the same in clique2.
        sep_marg = np.sum(clique1_dist[indexing_clique1])
        sep_dist[sepcells] = sep_marg

        # Set the shape of clique2 dist
        shape_clique2_dist = clique2_dist[indexing_clique2].shape
        alphas = [constant_alpha / np.prod(shape_clique2_dist)] * np.prod(shape_clique2_dist)
        # alphas = [constant_alpha] * np.prod(shape)  # TODO
        # Generate distribution of clique1 for the sep setting sepcell
        d = np.random.dirichlet(alphas).reshape(shape_clique2_dist)
        clique2_dist[indexing_clique2] = d * sep_marg
    return (clique2_dist, sep_dist)
    # return clique2_dist


def prob_dec(x, parameters, cliques, separators):
    """ Probability of numpy array x in a decomposable model.
    """
    log_prob = 0.0
    for c in cliques:
        # pick the values of x at the correct indices
        # Special case for cliques with only one node
        index = tuple(x[sorted(list(c))]) # the value of x is used as index in cont table for c.
        log_prob += np.log(parameters[c].item(index))

    for s in separators:
        if len(s) > 0:
            index = tuple(x[sorted(list(s))])
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
def locals_to_joint_prob_table(graph, parameters, levels):
    (cliques, separators, _, _, _) = trilearn.graph.decomposable.peo(graph)

    if len(cliques) == 1:
        separators = []
    elif len(cliques) > 1:
        separators = separators[1:]  # separators are counted from the second index

    no_levels = [len(l) for l in levels]
    table = np.zeros(np.prod(no_levels)).reshape(tuple(no_levels))
    # Iterate through al cells and set probabilities
    for cell in itertools.product(*levels):
        table[cell] = prob_dec(np.array(cell), parameters, cliques, separators)
    return table

def sample_prob_table(graph, levels, total_counts=1.0):
    local_tables = sample_hyper_consistent_parameters(graph, total_counts, levels)
    return locals_to_joint_prob_table(graph, local_tables, levels)


def sample_joint_prob_table(graph, levels, total_counts):
    local_tables = sample_hyper_consistent_parameters(graph, total_counts,
                                                      levels)
    table = locals_to_joint_prob_table(graph, local_tables, levels)
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