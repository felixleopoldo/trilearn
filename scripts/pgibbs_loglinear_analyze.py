from __future__ import unicode_literals

import json
import os
from os.path import basename

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from networkx.readwrite import json_graph
from pandas.plotting import autocorrelation_plot

import trilearn.auxiliary_functions
import trilearn.graph.graph as glib
import trilearn.graph.junction_tree as jtlib
import trilearn.graph.trajectory as mc
from trilearn.distributions import sequential_junction_tree_distributions as sjtd

np.set_printoptions(precision=2)

def main(data_filename, particles, alphas, betas, trajectory_length, graphfile, radius, output_directory,
         input_directory, pseudo_observations, burnin_end):

    filename = basename(data_filename)
    data = os.path.splitext(filename)[0]
    df = pd.read_csv(data_filename, sep=',', header=[0, 1])
    X = df.values.astype(int)
    n_levels = [int(a[1]) for a in list(df.columns)]
    levels = np.array([range(l) for l in n_levels])
    sample_size = df.shape[0]
    p = df.shape[1]

    radii = None
    if radius is None:
        radii = [p]
    else:
        radii = radius

    df = pd.DataFrame()
    for N_i, N in enumerate(particles):
        for T_i, T in enumerate(trajectory_length):
            for radius_i, radius in enumerate(radii):
                for alpha_i, alpha in enumerate(alphas):
                    for beta_i, beta in enumerate(betas):
                        for pseudo_i, pseudo_obs in enumerate(pseudo_observations):
                            cache = {}
                            # Load graph trajectory
                            seqdist_tmp = sjtd.LogLinearJTPosterior(X,
                                                                    pseudo_obs,
                                                                    levels,
                                                                    cache)

                            if graphfile:
                                with open(graphfile) as data_file:
                                    graph_json = json.load(data_file)
                                graph_true = json_graph.node_link_graph(graph_json)

                            pmcmc = "_"+str(seqdist_tmp)+"_T_"+str(T)+"_N_"+str(N)
                            pmcmc += "_alpha_"+str(alpha)
                            pmcmc += "_beta_"+str(beta)+'_radius_'+str(radius)
                            data_filename = data + pmcmc

                            # run_time = np.loadtxt(input_directory +
                            #                       "/"+data_filename +
                            #                       "_times.txt")

                            title = r"$n="+str(X.shape[0]) + \
                                    r", T="+str(T) + \
                                    r", N="+str(N) + \
                                    r", \alpha="+str(alpha) + \
                                    r", \beta="+str(beta) + \
                                    r", \rho=" + str(radius) + \
                                    r", \lambda=" + str(seqdist_tmp.cell_alpha) + \
                                    r"$" + \
                                    ", burn-in=0-" + str(burnin_end)
                            # ",\n run time=" + str(int(run_time/60)) + " min." + \
                            #print title

                            graph_traj = mc.Trajectory()
                            graph_traj.set_sequential_distribution(seqdist_tmp)
                            graph_traj.read_file(input_directory+"/"+data_filename +
                            "_graphs.txt")

                            with open(input_directory+"/"+data_filename +
                                      "_graphs.txt") as data_file:
                                js_graphs = json.load(data_file)
                            graphs = [json_graph.node_link_graph(js_G)
                            for js_G in js_graphs["trajectory"]]
                            p = graphs[0].order()

                            #run_time = np.loadtxt(input_directory +
                            #                      "/"+data_filename +
                            #                      "_times.txt")
                            # Calculate junction tree scores
                            graph_log_score_traj = np.array([None for _ in range(T)])
                            num_eqv_trees = np.array([None for _ in range(T)])
                            junction_trees = np.array([None for _ in range(T)])
                            for t in range(T):
                                JT = glib.junction_tree(graphs[t])
                                cliques = JT.nodes()
                                seps = jtlib.separators(JT)
                                graph_log_score_traj[t] = seqdist_tmp.ll(graphs[t])
                                num_eqv_trees[t] = jtlib.log_n_junction_trees(JT, seps)
                                junction_trees[t] = JT

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
                            tmp["lambda"] = pseudo_obs
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


    for N_i, N in enumerate(particles):
        for T_i, T in enumerate(trajectory_length):
            for radius_i, radius in enumerate(radii):
                for alpha_i, alpha in enumerate(alphas):
                    for beta_i, beta in enumerate(betas):
                        for pseudo_i, pseudo_obs in enumerate(pseudo_observations):
                            seqdist_tmp = sjtd.LogLinearJTPosterior(X,
                                                                    pseudo_obs,
                                                                    levels,
                                                                    cache)
                            pmcmc = "_"+str(seqdist_tmp)+"_T_"+str(T)+"_N_"+str(N)
                            pmcmc += "_alpha_"+str(alpha)
                            pmcmc += "_beta_"+str(beta)+'_radius_'+str(radius)
                            data_filename = data + pmcmc
                            filename_prefix = output_directory+"/" + \
                                               data_filename
                            #print filename_prefix

                            title = r"$n="+str(X.shape[0]) + \
                                    r", T="+str(T) + \
                                    r", N="+str(N) + \
                                    r", \alpha="+str(alpha) + \
                                    r", \beta="+str(beta) + \
                                    r", \rho=" + str(radius) + \
                                    r", \lambda=" + str(seqdist_tmp.cell_alpha) + \
                                    r"$" + \
                                    ", burn-in=0-" + str(burnin_end)
                                    # ",\n run time=" + str(int(run_time/60)) + " min." + \
                            #print title

                            tmp = df[(df["N"] == N) &
                                     (df["T"] == T) &
                                     (df["lambda"] == pseudo_obs) &
                                     (df["alpha"] == alpha) &
                                     (df["beta"] == beta) &
                                     (df["radius"] == radius)]

                            tmp_count = tmp["graph_tuple"].value_counts(normalize=True)
                            tmp_count.rename("Graph_counts")

                            #print "Empirical dist (top 1)"
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
                                    plt.yticks(rotation=0)
                                    plt.savefig(filename_prefix + "_top_"+str(i)+"_emp.eps",
                                                format="eps", bbox_inches='tight', dpi=100)
                                plt.clf()
                                with sns.axes_style("white"):
                                    sns.heatmap(heatmap, mask=mask, annot=False,
                                                cmap="Blues",
                                                xticklabels=range(1, p+1),
                                                yticklabels=range(1, p+1),
                                                vmin=0.0, vmax=1.0, square=True,
                                                cbar=True)
                                    plt.yticks(rotation=0)
                                    plt.savefig(filename_prefix + "_top_"+str(i)+"_emp_cbar.eps",
                                                format="eps", bbox_inches='tight', dpi=100)

                                i += 1

                            #print "Edge heatmap"
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
                                plt.yticks(rotation=0)
                                plt.savefig(filename_prefix+"_edge_heatmap_cbar.eps",
                                            format="eps", bbox_inches='tight', dpi=300)
                            plt.clf()
                            with sns.axes_style("white"):
                                ax = sns.heatmap(heatmap, mask=mask, annot=False,
                                                 cmap="Blues",
                                                 xticklabels=range(1, p+1),
                                                 yticklabels=range(1, p+1),
                                                 vmin=0.0, vmax=1.0, square=True,
                                                 cbar=False)
                                plt.yticks(rotation=0)
                                plt.savefig(filename_prefix+"_edge_heatmap.eps",
                                            format="eps", dpi=100, bbox_inches='tight')

                            #print "MAP graph"
                            ll_sorted_df = tmp.sort_values(by="log_likelihood",
                                                           ascending=False)[:1]
                            for row in ll_sorted_df.iterrows():
                                mat_tmp = nx.to_numpy_matrix(glib.tuple_to_graph(row[1]["graph_tuple"]))
                                plt.clf()
                                trilearn.auxiliary_functions.plot_matrix(mat_tmp,
                                                                         filename_prefix + "_MAP",
                                                "png",
                                                                         title="MAP graph")
                                # print glib.tuple_to_graph(row[1]["graph_tuple"]).edges()
                                # print row[1]["log_likelihood"]

                            ## Plot log-likelihood
                            plt.clf()
                            tmp["log_likelihood"].plot()
                            if graphfile:
                                tmp["log_likelihood_true"].plot()
                            plt.savefig(filename_prefix + "_pgibbs_log-likelihood.eps",
                                        format="eps", dpi=300)

                            # Plot auto correlation of number of edges
                            plt.clf()
                            with sns.axes_style("white"):
                                autocorrelation_plot(tmp["graph_size"])
                            plt.savefig(filename_prefix + "_pgibbs_autocorr_size.eps",
                                        format="eps", dpi=300)

                            # Auto correlation maximal clique
                            plt.clf()
                            autocorrelation_plot(tmp["maximal_clique"])
                            plt.savefig(filename_prefix + "_pgibbs_autocorr_maxl_clique.eps",
                                        format="eps", dpi=300)

    print "wrote"
    print output_directory + "/" +  \
        data_filename+"_edge_heatmap.png"
    print output_directory + "/" + \
        data_filename+"_tree_scores.png"
    print output_directory + "/" + \
        data_filename+"_graph_scores.png"
    print output_directory + "/" + \
        data_filename+"_MAP.png"
    print output_directory + "/" + data_filename + \
        "_results.json"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--data_filename',
        required=True,
        help="Filename of dataset"
    )
    parser.add_argument(
        '-N', '--particles',
        type=int, required=True,
        nargs='+', help="Number of SMC particles"
    )
    parser.add_argument(
        '-a', '--alphas',
        type=float, required=False, default=[0.5],
        nargs='+',
        help="Parameter for the junction tree expander"
    )
    parser.add_argument(
        '-b', '--betas',
        type=float, required=False, default=[0.5],
        nargs='+',
        help="Parameter for the junction tree expander"
    )
    parser.add_argument(
        '-M', '--trajectory_length',
        type=int, required=True,
        nargs='+',
        help="Number of Gibbs samples"
    )
    parser.add_argument(
        '-g', '--graphfile',
        required=False,
        help="The true graph in json-format"
    )
    parser.add_argument(
        '-r', '--radius',
        type=int, required=False,
        nargs='+',
        help="The search neighborhood radius for the Gibbs sampler"
    )
    parser.add_argument(
        '-o', '--output_directory',
        required=False, default=".",
        help="Output directory"
    )
    parser.add_argument(
        '-i', '--input_directory',
        required=False, default=".",
        help="Input directory"
    )
    parser.add_argument(
        '--pseudo_observations',
        type=float, required=False, default=[1.0],
        nargs='+',
        help="Total number of pseudo observations"
    )
    # parser.add_argument(
    #     '--n_levels', type=int, required=True,
    #     nargs='+',
    #     help="Number of levels for each variable"
    # )
    parser.add_argument(
        '--burnin_end',
        type=int, required=False, default=0,
        help="Burn-in ends here. default=0"
    )
    # parser.add_argument(
    #     '--data_header',
    #     type=int, default=None,
    #     help="Set to 0 if the data contains a header with names of the columns"
    # )

    args = parser.parse_args()
    main(**args.__dict__)