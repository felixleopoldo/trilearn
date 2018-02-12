from __future__ import unicode_literals

import json
import os
from os.path import basename

import matplotlib
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from networkx.readwrite import json_graph
from pandas.plotting import autocorrelation_plot

import trilearn.auxiliary_functions
import trilearn.graph.graph as glib
import trilearn.graph.junction_tree as jtlib
from trilearn.distributions import sequential_junction_tree_distributions as sjtd

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=8)
matplotlib.rc('ytick', labelsize=8)


np.set_printoptions(precision=2)

def trajectory_to_datafram():
    pass


def main(data_filename, n_particles, trajectory_length, radius, alphas, betas,
         graphfile, precmat, burnin_end, input_directory, output_directory):
    filename = basename(data_filename)
    data = os.path.splitext(filename)[0]
    X = np.matrix(np.loadtxt(data_filename, delimiter=','))
    sample_size = X.shape[0]
    p = X.shape[1]
    SS = X.T * X
    delta = 1.0
    D = np.identity(p)
    S = SS / sample_size
    radii = None
    if radius is None:
        radii = [p]
    else:
        radii = radius

    df = pd.DataFrame()
    for N_i, N in enumerate(n_particles):
        for T_i, T in enumerate(trajectory_length):
            for radius_i, radius in enumerate(radii):
                for alpha_i, alpha in enumerate(alphas):
                    for beta_i, beta in enumerate(betas):
                        cache = {}
                        seqdist_tmp = sjtd.GGMJTPosterior()
                        seqdist_tmp.init_model(X, D, delta, cache)
                        if graphfile:
                            with open(graphfile) as data_file:
                                graph_json = json.load(data_file)
                            graph_true = json_graph.node_link_graph(graph_json)
                            print "True graph size: " +str(graph_true.size())

                        # Load graph trajectory
                        pmcmc = "_"+str(seqdist_tmp)+"_T_"+str(T)+"_N_"+str(N)
                        pmcmc += "_alpha_"+str(alpha)
                        pmcmc += "_beta_"+str(beta)+'_radius_'+str(radius)
                        data_filename = data + pmcmc

                        # run_time = np.loadtxt(input_directory +
                        #                       "/"+data_filename +
                        #                       "_times.txt")

                        title = r"$T="+str(T)+", N="+str(N) + \
                                r", \alpha="+str(alpha) + \
                                r", \beta="+str(beta)+r", \rho="+str(radius)+r"$"
                                #print title
                        with open(input_directory+"/"+data_filename +
                                  "_graphs.txt") as data_file:
                            js_graphs = json.load(data_file)

                        graphs = [json_graph.node_link_graph(js_G)
                                  for js_G in js_graphs["trajectory"]]
                        p = graphs[0].order()

                        # Calculate junction tree scores
                        graph_log_score_traj = np.array([None for _ in range(T)])
                        num_eqv_trees = np.array([None for _ in range(T)])
                        #junction_trees = np.array([None for _ in range(T)])
                        for t in range(T):
                            JT = glib.junction_tree(graphs[t])
                            cliques = JT.nodes()
                            seps = jtlib.separators(JT)
                            graph_log_score_traj[t] = seqdist_tmp.ll(graphs[t])
                            num_eqv_trees[t] = jtlib.log_n_junction_trees(JT, seps)
                            #junction_trees[t] = JT

                        tmp = {}
                        tmp["log_likelihood"] = pd.Series(graph_log_score_traj)
                        tmp["graph"] = graphs
                        tmp["graph_tuple"] = [glib.graph_to_tuple(g1)
                                              for g1 in graphs]
                        tmp["graph_edges"] = [g1.edges() for g1 in graphs]
                        tmp["index"] = range(T)
                        tmp["N"] = N
                        tmp["T"] = T
                        tmp["alpha"] = alpha
                        tmp["beta"] = beta
                        tmp["radius"] = radius
                        tmp["adj_mat"] = [nx.to_numpy_matrix(g1)
                                          for g1 in graphs]
                        tmp["graph_size"] = [g1.size() for g1 in graphs]
                        tmp["maximal_clique"] = [nx.chordal_graph_treewidth(g1)+1
                                                 for g1 in graphs]
                        tmp["log_no_junction_trees"] = num_eqv_trees
                        if graphfile:
                            tmp["log_likelihood_true"] = seqdist_tmp.ll(graph_true)
                        tmp_df = pd.DataFrame(tmp)
                        df = df.append(tmp_df)

    for N_i, N in enumerate(n_particles):
        for T_i, T in enumerate(trajectory_length):
            for radius_i, radius in enumerate(radii):
                for alpha_i, alpha in enumerate(alphas):
                    for beta_i, beta in enumerate(betas):
                        seqdist_tmp = sjtd.GGMJTPosterior()
                        seqdist_tmp.init_model(X, D, delta, {})
                        # Load graph trajectory
                        pmcmc = "_"+str(seqdist_tmp)+"_T_"+str(T)+"_N_"+str(N)
                        pmcmc += "_alpha_"+str(alpha)
                        pmcmc += "_beta_"+str(beta)+'_radius_'+str(radius) #+ "_burnin_"+str(burnin_end)
                        data_filename = data + pmcmc
                        filename_prefix = output_directory+"/" + \
                            data_filename

                        title = r"$T="+str(T)+", N="+str(N) + \
                                r", \alpha="+str(alpha) + \
                                r", \beta="+str(beta)+r", \rho="+str(radius)+r"$"

                        tmp = (df[(df["N"] == N) &
                                  (df["T"] == T) &
                                  (df["alpha"] == alpha) &
                                  (df["beta"] == beta) &
                                  (df["radius"] == radius)])[burnin_end:]

                        tmp_count = tmp["graph_tuple"].value_counts(normalize=True)
                        tmp_count.rename("Graph_counts")

                        # Empirical dist (top 1)
                        i = 1
                        for tmp1 in tmp_count[:1].iteritems():
                            heatmap = nx.to_numpy_matrix(glib.tuple_to_graph(tmp1[0]))
                            plt.clf()
                            mask = np.zeros_like(heatmap)
                            mask[np.triu_indices_from(mask)] = True
                            with sns.axes_style("white"):
                                sns.heatmap(heatmap, mask=mask, annot=False,
                                            cmap="Blues",
                                            xticklabels=range(1, p+1),
                                            yticklabels=range(1, p+1),
                                            vmin=0.0, vmax=1.0, square=True,
                                            cbar=False)
                                plt.xticks(rotation=90)
                                plt.yticks(rotation=0)
                                plt.savefig(filename_prefix + "_top_"+str(i)+"_emp.eps",
                                            format="eps", bbox_inches='tight', dpi=300)
                            plt.clf()
                            with sns.axes_style("white"):
                                sns.heatmap(heatmap, mask=mask, annot=False,
                                            cmap="Blues",
                                            xticklabels=range(1, p+1),
                                            yticklabels=range(1, p+1),
                                            vmin=0.0, vmax=1.0, square=True,
                                            cbar=True)
                                plt.xticks(rotation=90)
                                plt.yticks(rotation=0)
                                plt.savefig(filename_prefix + "_top_"+str(i)+"_emp_cbar_"+str(burnin_end)+".eps",
                                            format="eps", bbox_inches='tight', dpi=300)

                            i += 1
                            # print str(tmp1[1]) + ": " + str(glib.tuple_to_graph(tmp1[0]).edges())

                        # Edge heatmap
                        heatmap = tmp["adj_mat"].sum() / len(tmp)
                        plt.clf()
                        mask = np.zeros_like(heatmap)
                        mask[np.triu_indices_from(mask)] = True
                        with sns.axes_style("white"):
                            sns.heatmap(heatmap, mask=mask, annot=False,
                                        cmap="Blues",
                                        xticklabels=range(1, p+1),
                                        yticklabels=range(1, p+1),
                                        vmin=0.0, vmax=1.0, square=True,
                                        cbar=True)
                            plt.xticks(rotation=90)
                            plt.yticks(rotation=0)
                            plt.savefig(filename_prefix+"_edge_heatmap_cbar_"+str(burnin_end)+".eps",
                                        format="eps", bbox_inches='tight', dpi=300)
                        plt.clf()
                        with sns.axes_style("white"):
                            ax = sns.heatmap(heatmap, mask=mask, annot=False,
                                             cmap="Blues",
                                             xticklabels=range(1, p+1),
                                             yticklabels=range(1, p+1),
                                             vmin=0.0, vmax=1.0, square=True,
                                             cbar=False)
                            plt.xticks(rotation=90)
                            plt.yticks(rotation=0)
                            plt.savefig(filename_prefix+"_edge_heatmap_"+str(burnin_end)+".eps",
                                        format="eps", dpi=300, bbox_inches='tight')

                        # MAP graph
                        ll_sorted_df = tmp.sort_values(by="log_likelihood", ascending=False)[:1]
                        for row in ll_sorted_df.iterrows():
                            mat_tmp = nx.to_numpy_matrix(glib.tuple_to_graph(row[1]["graph_tuple"]))
                            plt.clf()
                            trilearn.auxiliary_functions.plot_matrix(mat_tmp,
                                                                     filename_prefix + "_MAP",
                                            "eps",
                                                                     title="MAP graph")
                            # print glib.tuple_to_graph(row[1]["graph_tuple"]).edges()
                            # print row[1]["log_likelihood"]

                        # Plot log-likelihood
                        plt.clf()
                        tmp["log_likelihood"].plot()
                        if graphfile:
                            tmp["log_likelihood_true"].plot()
                        plt.savefig(filename_prefix + "_pgibbs_log-likelihood_"+str(burnin_end)+".eps",
                                    format="eps", dpi=300)

                        # Plot auto correlation of number of edges
                        plt.clf()
                        with sns.axes_style("white"):
                            autocorrelation_plot(tmp["graph_size"])
                        plt.savefig(filename_prefix + "_pgibbs_autocorr_size_"+str(burnin_end)+".eps",
                                    format="eps", dpi=300)

                        # Plot number of edges
                        plt.clf()
                        tmp["graph_size"].plot()
                        plt.savefig(filename_prefix + "_pgibbs_size_"+str(burnin_end)+".eps",
                                    format="eps", dpi=300)


                        # Auto correlation maximal clique
                        plt.clf()
                        autocorrelation_plot(tmp["maximal_clique"])
                        plt.savefig(filename_prefix + "_pgibbs_autocorr_maxl_clique_"+str(burnin_end)+".eps",
                                    format="eps", dpi=300)

                       # # Estimate omega with the posterior mean
                        # omega = np.zeros((p, p))
                        # for t in range(burnin_end, T):
                        #     graph = graphs[t]
                        #     omega += df.g_wishart_posterior_mean(graph, SS,
                        #                                          sample_size,
                        #                                          D, delta)
                        # omega = omega / (T-burnin_end)

                        # Compare precision matrices if exist
                        # if precmat:
                        #     true_omega = np.matrix(np.loadtxt(precmat,
                        #                                       delimiter=','))
                        #     map_est = df.g_wishart_posterior_mean(map_graph,
                        #                                            SS,
                        #                                            sample_size,
                        #                                            D, delta)

                        #     results = {}
                        #     results["L1(Omega_ML)"] = aux.l1_loss(S.I, true_omega)
                        #     results["L1(Omega_MAP_graph)"] = aux.l1_loss(map_est,
                        #                                                  true_omega)
                        #     results["L1(post_mean)"] = aux.l1_loss(omega,
                        #                                            true_omega)

                        #     results["L1(prec_glasso)"] = aux.l1_loss(gl.get_precision(),
                        #                                              true_omega)
                        #     results["L1(prec_glasso_cv)"] = aux.l1_loss(glcv.get_precision(),
                        #                                     true_omega)
                        #     results["L2(Omega_ML)"] = aux.l2_loss(S.I, true_omega)
                        #     results["L2(Omega_MAP_graph)"] = aux.l2_loss(map_est, true_omega)
                        #     results["L2(post_mean)"] = aux.l2_loss(omega,
                        #                                            true_omega)
                        #     results["L2(prec_glasso)"] = aux.l2_loss(gl.get_precision(), true_omega)
                        #     results["L2(prec_glasso_cv)"] = aux.l2_loss(glcv.get_precision(), true_omega)

                        #     print results
                        #     print
                        #     with open(output_directory + "/" + data_filename +
                        #               "_results.json",
                        #               'w') as outfile:
                        #         json.dump(results, outfile)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Generates analytics for the Markov chain of decomposable graphs generated "
                                     "by particle Gibbs.")

    parser.add_argument(
        '-f', '--data_filename',
        required=True,
        help="Filename of dataset")
    parser.add_argument(
        '-N', '--n_particles',
        type=int, required=True, nargs='+',
        help="Number of SMC particles")
    parser.add_argument(
        '-a', '--alphas', default=[0.5],
        type=float, required=False, nargs='+',
        help="Parameter for the junction tree expander")
    parser.add_argument(
        '-b', '--betas', default=[0.5],
        type=float, required=False, nargs='+',
        help="Parameter for the junction tree expander")
    parser.add_argument(
        '-M', '--trajectory_length',
        type=int, required=True, nargs='+',
        help="Number of Gibbs samples")
    parser.add_argument(
        '-g', '--graphfile',
        required=False,
        help="The true graph in json-format")
    parser.add_argument(
        '-p', '--precmat',
        required=False,
        help="The true precision matrix")
    parser.add_argument(
        '-r', '--radius',
        type=int, required=False, default=None, nargs='+',
        help="The search neighborhood radius for the Gibbs sampler")
    parser.add_argument(
        '-o', '--output_directory',
        required=False, default="./",
        help="Output directory")
    parser.add_argument(
        '-i', '--input_directory',
        required=False, default="./",
        help="Input directory")
    parser.add_argument(
        '--burnin_end', type=int, required=False, default=0,
        help="Burnin ends here")

    args = parser.parse_args()
    main(**args.__dict__)