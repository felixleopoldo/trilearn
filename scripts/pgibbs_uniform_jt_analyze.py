from networkx.readwrite import json_graph

import trilearn.graph.decomposable
import trilearn.graph.graph as glib
import trilearn.graph.junction_tree as jtlib
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


np.set_printoptions(precision=3)

parser = argparse.ArgumentParser()
parser.add_argument('-N', '--particles', type=int, required=True,
                    nargs='+', help="Number of SMC particles")
parser.add_argument('-a', '--alpha', type=float, required=True,
                    nargs='+', help="Parameter for the junction tree expander")
parser.add_argument('-b', '--beta', type=float, required=True,
                    nargs='+', help="Parameter for the junction tree expander")
parser.add_argument('-T', '--trajectory_length', type=int, required=True,
                    nargs='+', help="Number of Gibbs samples")
parser.add_argument('-r', '--radius', type=int, required=True,
                    nargs='+', help="The search neighborhood +"
                    "+radius for the Gibbs sampler")
parser.add_argument('-p', '--order', type=int, required=True,
                    help="The order of the underlying decomposable graph")
parser.add_argument('--input_directory', required=True,
                    help="Directory of Markov chain samples")
parser.add_argument('--output_directory', required=True, default='.',
                    help="Directory")
args = parser.parse_args()

for N in args.particles:
    for T in args.trajectory_length:
        for radius in args.radius:
            for alpha in args.alpha:
                for beta in args.beta:
                    filename = args.input_directory+"/" \
                               "uniform_jt_samples_p_"+str(args.order)+"_T_" \
                               + str(T)+"_N_"+str(N) \
                               + "_alpha_"+str(alpha) \
                               + "_beta_"+str(beta)+'_radius_'+str(radius)

                    with open(filename+"_graphs.txt") as data_file:
                        js_graphs = json.load(data_file)
                        graphs = [json_graph.node_link_graph(js_G)
                                  for js_G in js_graphs]
                        p = graphs[0].order()

                    jt_width_count = np.array([0.0 for _ in range(p)])
                    graph_max_clique_count = np.array([0.0
                                                       for _ in range(p+1)])
                    for t in range(T):
                        JT = trilearn.graph.decomposable.junction_tree(graphs[t])
                        separators = jtlib.separators(JT)
                        cliques = JT.nodes()
                        clique_sizes = [len(q) for q in JT.nodes()]

                        jt_width_count[max(clique_sizes)-1] += 1.0
                        mu = np.exp(jtlib.log_n_junction_trees(JT, separators))
                        graph_max_clique_count[max(clique_sizes)] += 1.0/mu

                    print "Tree width distribution"
                    print jt_width_count / T
                    print "Mean tree width"
                    print sum((jt_width_count / T) * range(p))

                    print "Graph maximal clique distribution"
                    print graph_max_clique_count / sum(graph_max_clique_count)

                    plt.bar(range(p),
                            jt_width_count / float(sum(jt_width_count)))
                    plt.title("Threewidth distribution under uniform " +
                              "sampling from T(" + str(args.order) + ").\n " +
                              "T="+str(T)+", N="+str(N) +
                              ", alpha="+str(alpha) +
                              ", beta=" + str(beta))
                    plt.xlabel('Three width')
                    plt.ylabel('Frequency')
                    plt.savefig(args.output_directory + "/treewidth.png")
                    plt.clf()

                    plt.bar(range(p+1),
                            graph_max_clique_count/float(sum(graph_max_clique_count)))
                    plt.title("Maximal clique distribution under uniform " +
                              "sampling from T(" + str(args.order) + ").\n " +
                              "T="+str(T)+", N="+str(N) +
                              ", alpha="+str(alpha) +
                              ", beta=" + str(beta))
                    plt.xlabel('Maximal clique size')
                    plt.ylabel('Frequency')
                    plt.savefig(args.output_directory + "/max_clique_size.png")
                    plt.clf()
