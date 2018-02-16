import json
import argparse

import networkx as nx
from networkx.readwrite import json_graph
import numpy as np

import trilearn.auxiliary_functions
import trilearn.graph.junction_tree
import trilearn.graph.junction_tree as jtlib
import trilearn.graph.christmas_tree_algorithm as jtexp
import trilearn.graph.graph as glib


def main(n_dim, alpha, beta, output_directory, seed, **args):
    if seed > 0:
        np.random.seed(seed)
    order = range(n_dim)
    np.random.shuffle(order)
    T = trilearn.graph.junction_tree.sample_junction_tree(order, alpha, beta)
    G = jtlib.graph(T)
    
    graph_name = "graph_p_"+str(len(order))
    graph_name += "_alpha_"+str(alpha)+"_beta_"+str(beta)
    jt_name = "junction_tree_p_"+str(len(order))
    jt_name += "_alpha_"+str(alpha)+"_beta_"+str(beta)
    
    with open(output_directory+"/"+graph_name+".json", 'w') as outfile:
        js_graph = json_graph.node_link_data(G)
        json.dump(js_graph, outfile)
    
    T1 = T.subgraph(T.nodes())
    
    glib.plot(G, output_directory+"/"+graph_name+".eps", layout="fdp")
    glib.plot(T1, output_directory+"/"+jt_name+".eps")
    
    hm_true = np.array(nx.to_numpy_matrix(G, nodelist=range(n_dim)))
    trilearn.auxiliary_functions.plot_matrix(hm_true,
                                             output_directory +"/" + graph_name +"_heatmap",
                   "eps",
                   "True graph")
    print "wrote"
    print output_directory+"/"+jt_name+".json"
    print output_directory+"/"+jt_name+".eps"
    print output_directory+"/"+graph_name+".json"
    print output_directory+"/"+graph_name+".eps"
    print output_directory+"/"+graph_name+"_heatmap.eps"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generates a random junction tree and the corresponding "
                    "decomposable graph using the Christmas tree algorithm.")
    parser.add_argument(
        '-a', '--alpha',
        type=float, required=False, default=0.5,
        help="Sparsity parameter"
    )
    parser.add_argument(
        '-b', '--beta',
        type=float, required=False, default=0.5,
        help="Sparsity parameter (probability of create isolated node in each iteration)."
    )
    parser.add_argument(
        '-p', '--n_dim',
        type=int, required=True,
        help="Number of nodes in the underlying graph.")
    parser.add_argument(
        '-o', '--output_directory',
        required=False, default="."
    )
    parser.add_argument(
        '-s', '--seed',
        type=int, required=False
    )
    args = parser.parse_args()
    main(**args.__dict__)