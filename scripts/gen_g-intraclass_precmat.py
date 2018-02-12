import os
import argparse
import json

from networkx.readwrite import json_graph
import numpy as np

from trilearn.distributions import g_intra_class

def main(graph_filename, sigma2, rho, output_directory):
    filename = os.path.basename(graph_filename)
    basename = os.path.splitext(filename)[0]

    with open(graph_filename) as data_file:
        js_graph = json.load(data_file)

    graph = json_graph.node_link_graph(js_graph)

    # Gen covariance
    intraclass_sigma = sigma2
    intraclass_rho = rho
    sigma = g_intra_class.cov_matrix(graph,
                                     intraclass_rho,
                                     intraclass_sigma)
    precmat = sigma.I
    filename_prefix = "intra-class_precmat_graph_" + basename + \
                      "_rho_"+str(rho)+"_sigma2_" + \
                      str(sigma2)
    filename_write = output_directory + "/" + filename_prefix + ".txt"

    np.savetxt(filename_write,
               precmat, delimiter=',', fmt="%f")
    print "wrote"
    print filename_write


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate so called graph intraclass precision matrix given a graph.")
    parser.add_argument('--rho', type=float, required=True,
                        help="Correlation coefficient")
    parser.add_argument('--sigma2', type=float, required=True,
                        help="Variance (diagonal values)")
    parser.add_argument('--graph_filename', required=True,
                        help="Graph json file")
    parser.add_argument('-o', '--output_directory', required=False, default=".",
                        help="Output directory")

    args = parser.parse_args()
    main(**args.__dict__)
