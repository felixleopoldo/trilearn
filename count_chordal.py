
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trilearn.smc as smc
from tabulate import tabulate
from tqdm import tqdm
from decimal import Decimal
from scipy.special import comb
import trilearn.distributions.sequential_junction_tree_distributions as seqdist

np.random.seed(1)

p = 50
N = 1000
T = 2

counts = np.zeros((p+1) * T).reshape(T, (p+1))
for t in tqdm(range(T)):
    counts[t,1:] = smc.est_n_dec_graphs(p, N, debug=False)

n_graphs = [2**comb(m, 2) for m in range(p+1)]
est_fracs = np.divide(counts, n_graphs)
est_fracs_means = est_fracs.mean(axis=0)
est_fracs_sterr = est_fracs.std(axis=0)
means = counts.mean(axis=0)
sterr = counts.std(axis=0)
n_graphs_true = np.array([0, 1, 2, 8, 61, 822, 18154, 617675, 30888596, 
                          2192816760, 215488096587, 28791414081916, 
                          5165908492061926, 1234777416771739141] + [0]*(p-13))

true_fracs = np.divide(n_graphs_true[:p+1], n_graphs)

sn = np.zeros(p+1)
for pp in range(1, p+1):
    sn[pp] = np.sum([comb(pp, r, exact=False) * (2 ** (r * (pp - r))) for r in range(1, pp + 1)])


# Plot of the true, estimated, and asymptotic numbers.
#bn = np.zeros(p+1)
#bn[12] = 4818917841228328
#bn[13] = 1167442027829857677
#print bn[12] / sn[12] # Missing factor 2
#print bn[13] / sn[13] # Missing factor 2
plt.semilogy(range(1, p+1), means[1:p+1], '-*', label="SMC", alpha=0.4)
plt.semilogy(range(1, p+1), sn[1:p+1], '-.', label="Asymptotic")
plt.semilogy(range(1, min(14, p+1)), n_graphs_true[1: min(14, p+1)], '-+', label="Exact")
plt.legend(shadow=False, fancybox=True)
plt.savefig("n_chordal_p"+str(p)+"_N_"+str(N)+"_T_"+str(T)+".eps", format="eps")

## Print latex table
table = [np.arange(p+1)] # 0
table.append(sn[:p+1]) # 1

table.append(n_graphs_true[:p+1]) # 2 
table.append(means[:p+1]) # 3
table.append(sterr[:p+1]) # 4

table.append(true_fracs[:p+1])
table.append(est_fracs_means[:p+1])
table.append(est_fracs_sterr[:p+1])

for row in range(1, p+1):
    print str(row) + "\t&\t" + \
          str('%.2E' % Decimal(table[1][row])) + "\t&\t" + \
          str('%.2E' % Decimal(table[2][row])) + "\t&\t" + \
          str('%.2E' % Decimal(table[3][row])) + "\t&\t" + \
          str('%.2E' % Decimal(table[4][row])) + " \\\\\n"

for row in range(1, p+1):
    print str(row) + "\t&\t" + \
          str('%.2E' % Decimal(table[5][row])) + "\t&\t" + \
          str('%.2E' % Decimal(table[6][row])) + "\t&\t" + \
          str('%.2E' % Decimal(table[7][row])) + " \\\\\n"


# Print number of junction trees
n_trees_true = [0, 1, 2, 10, 108, 2091]
n_trees_true_padded = np.array(n_trees_true + [0]*(p-5))

sd = seqdist.UniformJTDistribution(p) 
n_trees = np.zeros((p+1) * T).reshape(T, (p+1))
for t in tqdm(range(T)):
    n_trees[t,1:] = np.exp(smc.est_log_norm_consts(p, N, sd, 0.5, 0.5, 1, False))

n_trees_means = n_trees.mean(axis=0)
n_trees_sterr = n_trees.std(axis=0)

table = [np.arange(p+1)] # 0

table.append(n_trees_true_padded[:p+1]) # 1
table.append(n_trees_means[:p+1]) # 2
table.append(n_trees_sterr[:p+1]) # 3

true_tree_graph_frac = np.divide(np.array(n_trees_true_padded[:p+1], dtype=float), n_graphs_true[:p+1])
est_tree_graph_frac = np.divide(n_trees, counts)

est_tree_graph_frac_means = est_tree_graph_frac.mean(axis=0)
est_tree_graph_frac_sterr = est_tree_graph_frac.std(axis=0)

table.append(true_tree_graph_frac[:p+1]) # 4
table.append(est_tree_graph_frac_means[:p+1]) # 5
table.append(est_tree_graph_frac_sterr[:p+1]) # 6

for row in range(1, p+1):
    print str(row) + "\t&\t" + \
          str('%.2E' % Decimal(table[1][row])) + "\t&\t" + \
          str('%.2E' % Decimal(table[2][row])) + "\t&\t" + \
          str('%.2E' % Decimal(table[3][row])) + "\t&\t" + \
          " \\\\\n" 
          #str('%.2E' % Decimal(table[4][row])) + " \\\\\n"

for row in range(1, p+1):
    print str(row) + "\t&\t" + \
          str('%.2E' % Decimal(table[4][row])) + "\t&\t" + \
          str('%.2E' % Decimal(table[5][row])) + "\t&\t" + \
          str('%.2E' % Decimal(table[6][row])) + " \\\\\n"


plt.clf()
plt.semilogy(range(1,p+1), est_tree_graph_frac_means[1:], ".-")
plt.savefig("tree_graph_frac.eps",format="eps")
