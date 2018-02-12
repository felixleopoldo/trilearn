import json
import argparse

import numpy as np
import networkx as nx
from networkx.readwrite import json_graph

import chordal_learning.auxiliary_functions
import chordal_learning.graph.graph as libg
import chordal_learning.distributions.g_intra_class as gic


np.set_printoptions(precision=1)
# G-Intra class (AR(2))


def main(s2, rho, n_samples, n_dim, graph_dir, data_dir, precmat_dir, **args):
    G = nx.Graph()
    for i in range(n_dim):
        G.add_node(i, label=str(i+1))
    for i in range(n_dim-2):
        G.add_edges_from([(i, i+1), (i, i+2)])
    G.add_edge(n_dim-2, n_dim-1)
    X = gic.sample(G, rho, s2, n_samples).T
    c = gic.cov_matrix(G, rho, s2)

    hm_true = np.array(nx.to_numpy_matrix(G, nodelist=range(n_dim)))
    chordal_learning.auxiliary_functions.plot_matrix(hm_true, graph_dir + "/intraclass_p_" + str(n_dim) +
                   "_heatmap", "eps", "True graph")
    libg.plot_graph(G, graph_dir + "/intraclass_p_" +
                    str(n_dim)+"_graph.eps")
    np.savetxt(data_dir + "/intraclass_p_"+str(n_dim)+"_sigma2_"+str(s2) +
               "_rho_"+str(rho)+"_n_"+str(n_samples)+".csv", X, delimiter=',')
    np.savetxt(precmat_dir + "/intraclass_p_"+str(n_dim)+"_sigma2_"+str(s2) +
               "_rho_"+str(rho)+"_omega.txt", c.I, delimiter=',')

    with open(graph_dir +
              "/intraclass_p_"+str(n_dim)+".json", 'w') as outfile:
        js_graph = json_graph.node_link_data(G)
        json.dump(js_graph, outfile)

    print "Generated files:"
    print graph_dir + "/intraclass_p_"+str(n_dim)+"_heatmap.eps"
    print graph_dir + "/intraclass_p_"+str(n_dim)+"_graph.eps"
    print data_dir + "/intraclass_p_"+str(n_dim)+"_sigma2_"+str(s2) + \
        "_rho_"+str(rho)+"_n_"+str(n_samples)+".csv"
    print precmat_dir + "/intraclass_p_"+str(n_dim) + \
        "_sigma2_"+str(s2)+"_rho_"+str(rho)+"_omega.txt"
    print graph_dir + "/intraclass_p_"+str(n_dim)+".json"

    if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generates samples from a graph "
                                                 "intra-class model and saves the precision matrix.")
    parser.add_argument(
        '--sigma2',
        type=float, required=True,
        description='Variance'

    )
    parser.add_argument(
        '--rho',
        type=float, required=True,
        description="Correlation coefficient"

    )
    parser.add_argument(
        '-n', '--n_samples',
        type=int, required=True,
        description="Number of normal samples"
    )
    parser.add_argument(
        '-p', '--n_dim',
        type=int, required=True,
        descritption="Number of dimensions"
    )
    parser.add_argument(
        '--graph_dir',
        required=False, default=".",
        description="Directory where to save the graph file"
    )
    parser.add_argument(
        '--data_dir',
        required=False, default=".",
        description="Directory where to save the data file"
    )
    parser.add_argument(
        '--precmat_dir',
        required=False, default=".",
        description="Directory where to save the"
    )

    args = parser.parse_args()
    main(**__dict__)

