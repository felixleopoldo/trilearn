import os
from os.path import basename

import numpy as np
import json
import argparse
from networkx.readwrite import json_graph

import trilearn.graph.graph as libg
from trilearn.distributions import discrete_dec_log_linear as loglin


def main(graph_filename, parameters_filename, data_samples, output_directory, **args):
    param_filename = basename(parameters_filename)
    param_basename = os.path.splitext(param_filename)[0]

    filename = basename(graph_filename)
    graph_basename = os.path.splitext(filename)[0]
    graph_file = graph_filename

    with open(graph_file) as data_file:
        json_G = json.load(data_file)

    graph = json_graph.node_link_graph(json_G)
    (C, S, H, A, R) = libg.peo(graph)

    with open(parameters_filename) as data_file:
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

    tot_prob = 0.0
    table = None

    if len(S) > 1:
        table = loglin.full_prob_table(parameters, levels, C, S[1:])
    else:
        table = loglin.full_prob_table(parameters, levels, C, [])

    data = loglin.gen_multidim_data(table, 1000)

    print data.shape
    np.savetxt(output_directory+"/"+param_basename +
               "_n_"+str(data_samples)+".csv",
               data, delimiter=',', fmt="%i")

    print "wrote"
    print output_directory+"/"+param_basename +"_n_" + \
        str(data_samples)+".csv"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--graph_filename',
        required=True
    )
    parser.add_argument(
        '--output_directory',
        required=False, default="."
    )
    parser.add_argument(
        '--parameters_filename',
        required=True
    )
    parser.add_argument(
        '-n', '--data_samples',
        required=True
    )
    parser.add_argument(
        '-s', '--seed',
        type=int, required=True
    )

   main(**args.__dict__)