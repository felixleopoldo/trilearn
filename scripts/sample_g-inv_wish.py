import os

import json
from networkx.readwrite import json_graph
import numpy as np

import trilearn.auxiliary_functions
from trilearn.distributions import g_inv_wishart


np.set_printoptions(precision=1)


def main(graph_filename, output_directory, seed):

    if seed > 0:
        np.random.seed(seed)
    
    # Randomly generated graph
    filename = os.path.basename(graph_filename)
    basename = os.path.splitext(filename)[0]
    graph_file = graph_filename
    
    
    with open(graph_file) as data_file:
        json_G = json.load(data_file)
    
    G = json_graph.node_link_graph(json_G)
    p = G.order()
    b = p  # Degrees of freedom
    D = np.matrix(np.identity(p))  # Scale parameter
    sigma = g_inv_wishart.sample(G, b, D)
    
    trilearn.auxiliary_functions.plot_matrix(np.array(sigma.I),
                                             output_directory +"/G-invwish_G_" + basename +
                         "_b_" + str(b) +"_D_I", "eps", "Sigma inverse")
    
    np.savetxt(output_directory+"/G-invwish_G_" +
               basename + "_b_"+str(b)+"_D_I.txt",
               sigma, delimiter=',', fmt="%f")
    print "Generated files:"
    print output_directory+"/G-invwish_G_"+basename+"_b_"+str(b)+"_D_I.txt"
    print output_directory+"/G-invwish_G_"+basename+"_b_"+str(b)+"_D_I.eps"


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--graph_filename', required=True)
    parser.add_argument('-o', '--output_directory', required=False, default=".")
    parser.add_argument('-s', '--seed', type=int, required=False, default=None)
    args = parser.parse_args()

    args = parser.parse_args()
    main(**args.__dict__)