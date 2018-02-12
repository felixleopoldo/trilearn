import os

import numpy as np
import json
from networkx.readwrite import json_graph

import trilearn.graph.graph as libg
import trilearn.graph.junction_tree as libj
from trilearn.distributions import discrete_dec_log_linear as loglin


def main(graph_filename, n_levels, pseudo_obs, output_directory, seed, **args):

    if seed > 0:
        np.random.seed(seed)

    # Nodes should be enumerate 0,...,p-1
    # Levels are enumerated 0,...,k-1

    filename = os.path.basename(graph_filename)
    basename = os.path.splitext(filename)[0]
    #graph_file = graph_filename

    with open(graph_filename) as data_file:
        json_G = json.load(data_file)

    graph = json_graph.node_link_graph(json_G)
    p = graph.order()
    junctiontree = libg.junction_tree(graph)
    (C, S, H, A, R) = libj.peo(junctiontree)

    levels = np.array([range(l) for l in n_levels])

    parameters = loglin.gen_globally_markov_distribution(graph, pseudo_obs,
                                                         levels)

    parameters_flattened = {}
    parameters_flattened["no_levels"] = n_levels
    for key, val in parameters.iteritems():
        props = {}
        props["parameters"] = list(val.reshape(np.prod(val.shape)))
        props["clique_nodes"] = list(key)
        parameters_flattened[str(list(key))] = props

    parameters_filename = output_directory+"/"+basename+"_loglin_parameters_lambda_" \
                          + str(pseudo_obs) + ".json"

    with open(parameters_filename, 'w') as outfile:
        json.dump(parameters_flattened, outfile)

    print "wrote"
    print parameters_filename

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Sample parameters for a discrete log-linear "
                                                 "model which is Markov w.r.t. a graph.")
    parser.add_argument(
        '-s', '--seed',
        required=False,
        type=int
    )
    parser.add_argument(
        '-g', '--graph_filename',
        required=True
    )
    parser.add_argument(
        '--pseudo_obs',
        type=float, required=False, default="1.0",
        help="Total number of pseudo observations to "
             "be distributed uniformly to each cell."
    )
    parser.add_argument(
        '--n_levels',
        type=int, nargs="+", required=True,
        help="Number of levels for each cell. E.g. for 4 variables each "
             "taking values 0 or 1 would be: 2 2 2 2 "
        )
    parser.add_argument(
        '--output_directory',
        required=False, default="."
    )

    args = parser.parse_args()
    main(**args.__dict__)
