import json
import os

from networkx.readwrite import json_graph
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from trilearn.smc_functions import *

data = "intra_class_sigma2_1.0_rho_0.9_n_300_p_6"
filename = data+"_smc_N_2000_alpha_0.3_beta_0.9"

with open(filename+".txt") as data_file:
    js_graphs = json.load(data_file)
w = np.loadtxt(filename+"_weights.txt",delimiter=',')
N = len(js_graphs)
p = json_graph.node_link_graph(js_graphs[0]).order()
print w

plt.hist(w)
plt.show()

graphs = {}
for i in range(N):
    graph = json_graph.node_link_graph(js_graphs[i])
    weight = w[i, p-1]

    graph_tuple = nx.to_numpy_matrix(graph, nodelist=range(p))
    graph_tuple = np.array(graph_tuple, dtype=int)
    graph_tuple = graph_tuple.flatten()
    graph_tuple = graph_tuple.tolist()
    graph_tuple = tuple(graph_tuple)

    if not graph_tuple in graphs:
        graphs[graph_tuple] = []
    graphs[graph_tuple].append(weight)

# print graphs
print "Approximated distribution"
ordered_graphs = sorted(graphs.iteritems(), key=lambda (k,v): (sum(v), k), reverse=True)
#print ordered_graphs
tot = sum( [sum(we) for (k,we) in ordered_graphs ])
topgraphs = 0
for key, value in ordered_graphs:
    if topgraphs > 0: break
    matrix = np.matrix(list(key)).reshape((p,p))
    topgraphs += 1
    print "%s: %s" % (matrix, sum(value)/tot)
    plotGraph( nx.to_networkx_graph(matrix, create_using = nx.Graph()), filename+"_MAP_"+str(topgraphs)+".eps")

hm = np.zeros(p*p).reshape( (p,p) )

for graph, weights in graphs.iteritems():
    # print graph, weights
    tmp = graph #tuple
    tmp = list(tmp)
    matrix = np.matrix(tmp)
    matrix.shape = (p,p)
    hm += matrix * sum(weights)

plot_matrix(hm,filename+"_smc_heatmap","eps",r"Posterior edge heat map for $p$="+str(p))

# OMega inference
X = np.matrix(np.loadtxt(data+".data",delimiter=',')).T
n_normal_samples = X.shape[1]
SS = X * X.T
S = SS/n_normal_samples
delta = 1.0
D = np.identity(p) #sigma * delta

omega = np.zeros((p, p))
for graph_tuple, weights in graphs.iteritems():
    tmp = graph_tuple #tuple
    tmp = list(tmp)
    matrix = np.matrix(tmp)
    matrix.shape = (p,p)
    graph = nx.from_numpy_matrix(matrix)
    omega += sum(weights) * g_wishart_posterior_mean(graph, SS, n_normal_samples, D, delta)

subset = np.ix_(range(6),range(6))
print "omega pmcmc posterior mean"
print omega[subset]
print "omega.I pmcmc posterior mean"
print np.matrix(omega).I[subset]
print "S"
print S[subset]
print "S.I"
print S.I[subset]
plot_matrix(np.abs(omega), filename+"_omega_heatmap","eps",r"Posterior heat map for $\bf{\Theta}$")
plot_matrix(np.abs(np.array(S.I)),data+"_S_inv","eps",r"$\bf{S}^{-1}$ heat map")
plot_matrix(np.abs(np.array(S)),data+"_S","eps",r"$\bf S$ heat map")

if os.path.isfile(data + "_permutation.txt"):
    perm = np.loadtxt(data + "_permutation.txt",delimiter=',')
    perm = [int(i) for i in perm ]
    print perm
    inv_perm = [0]*p
    for i,j in enumerate(perm): inv_perm[j] = i
    plot_matrix(hm[inv_perm][:,inv_perm],filename+"_smc_heatmap_invperm","eps","Posterior heat map for p="+str(p))

