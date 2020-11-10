
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trilearn.smc as smc
from tabulate import tabulate
from tqdm import tqdm
from decimal import Decimal
from scipy.special import comb
import trilearn.distributions.sequential_junction_tree_distributions as seqdist

p = 50

n_chordal_true = np.array([0, 1, 2, 8, 61, 822, 18154, 617675, 30888596, 
                          2192816760, 215488096587, 28791414081916, 
                          5165908492061926, 1234777416771739141] + [0]*(p-13))

n_undirected_graphs = [2**comb(m, 2) for m in range(p+1)]
frac_undirected_chordal_true = np.divide(n_chordal_true[:p+1], n_undirected_graphs)

print("True number of chordal graphs")
print(n_chordal_true)
print("True number of undirected graphs")
print(n_undirected_graphs)
print("True number of chordal graph per undirected graph")
print(frac_undirected_chordal_true)
# Asymptotic number of decomposable graphs
sn = np.zeros(p+1)
for pp in range(1, p+1):
    sn[pp] = np.sum([comb(pp, r, exact=False) * (2 ** (r * (pp - r))) for r in range(1, pp + 1)])

print("Asymptotic number of chordal graphs")
print(sn)

N = 10000
T = 10
chordal_df = pd.DataFrame(columns=["order", "n_chordal_est", "seed"])
n_chordal_est = np.zeros((p+1) * T).reshape(T, (p+1))
for t in tqdm(range(T)):
    filename = Path("n_chordal_est_N={N}_p={p}_seed={seed}.csv".format(seed=t,N=N,p=p))
    if not filename.is_file():
        df = pd.DataFrame(columns=["order", "n_chordal_est", "seed"])
        np.random.seed(t)
        n_chordal_est[t,1:] = smc.est_n_dec_graphs(p, N, debug=False)
        df["n_chordal_est"] = n_chordal_est[t,1:]
        df["seed"] = t
        df["order"] = range(1,p+1)

        df.to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename)
        n_chordal_est[t,1:] = df["n_chordal_est"].values

    chordal_df = chordal_df.append(df)


#print(n_chordal_est)
#print(chordal_df.groupby(["order"]).mean())
#print(chordal_df.groupby(["order"]).std())
frac_undirected_chordal_est = np.divide(n_chordal_est, n_undirected_graphs)
frac_undirected_chordal_est_means = frac_undirected_chordal_est.mean(axis=0)
frac_undirected_chordal_est_stds = frac_undirected_chordal_est.std(axis=0)
n_chordal_est_means = n_chordal_est.mean(axis=0)
n_chordal_est_stds = n_chordal_est.std(axis=0)


# Plot of the true, estimated, and asymptotic numbers.
plt.semilogy(range(1, p+1), n_chordal_est_means[1:p+1], '-*', label="SMC", alpha=0.4)
plt.semilogy(range(1, p+1), sn[1:p+1], '-.', label="Asymptotic")
plt.semilogy(range(1, min(14, p+1)), n_chordal_true[1: min(14, p+1)], '-+', label="Exact")
plt.legend(shadow=False, fancybox=True)
plt.savefig("n_chordal_p"+str(p)+"_N_"+str(N)+"_T_"+str(T)+".eps", format="eps")

## Print latex table
table_chordal = [np.arange(p+1)] # 0
table_chordal.append(sn[:p+1]) # 1

table_chordal.append(n_chordal_true[:p+1]) # 2 
table_chordal.append(n_chordal_est_means[:p+1]) # 3
table_chordal.append(n_chordal_est_stds[:p+1]) # 4

table_chordal.append(frac_undirected_chordal_true[:p+1])
table_chordal.append(frac_undirected_chordal_est_means[:p+1])
table_chordal.append(frac_undirected_chordal_est_stds[:p+1])

print("Number of chordal graphs")
for row in range(1, p+1):
    print str(row) + "\t&\t" + \
          str('%.2E' % Decimal(table_chordal[2][row])) + "\t&\t" + \
          str('%.2E' % Decimal(table_chordal[3][row])) + "\t&\t" + \
          str('%.2E' % Decimal(table_chordal[4][row])) + " & $\\times 10^{}$\\\\\n"
          #str('%.2E' % Decimal(table_chordal[1][row])) + "\t&\t" + \
print("Fraction of undirected graphs that are chordal")
for row in range(1, p+1):
    print str(row) + "\t&\t" + \
          str('%.2E' % Decimal(table_chordal[5][row])) + "\t&\t" + \
          str('%.2E' % Decimal(table_chordal[6][row])) + "\t&\t" + \
          str('%.2E' % Decimal(table_chordal[7][row])) + " & $\\times 10^{}$\\\\\n"


print("Counting junction trees")
# Print number of junction trees
n_trees_true = [0, 1, 2, 10, 108, 2091]
n_trees_true_padded = np.array(n_trees_true + [0]*(p-5))

sd = seqdist.UniformJTDistribution(p) 
n_trees_est = np.zeros((p+1) * T).reshape(T, (p+1))
n_trees_df = pd.DataFrame(columns=["order", "n_trees_est", "seed"])
for t in tqdm(range(T)):
    filename = Path("n_trees_est_N={N}_p={p}_seed={seed}.csv".format(seed=t,N=N,p=p))
    if not filename.is_file():
        np.random.seed(t)
        df = pd.DataFrame(columns=["order", "n_trees_est", "seed"])
        n_trees_est[t,1:] = np.exp(smc.est_log_norm_consts(p, N, sd, 0.5, 0.5, 1, False))
        df["order"] = range(1,p+1)
        df["n_trees_est"] = n_trees_est[t, 1:]
        df["seed"] = t
        #print(t)
        df.to_csv(filename, index=False)
    else:
        df = pd.read_csv(filename)
        n_trees_est[t,1:] = df["n_trees_est"].values

    n_trees_df = n_trees_df.append(df)

#print(n_trees_df.groupby(["order"]).mean())
#print(n_trees_df.groupby(["order"]).std())


n_trees_means = n_trees_est.mean(axis=0)
n_trees_sterr = n_trees_est.std(axis=0)

table_tree = [np.arange(p+1)] # 0

table_tree.append(n_trees_true_padded[:p+1]) # 1
table_tree.append(n_trees_means[:p+1]) # 2
table_tree.append(n_trees_sterr[:p+1]) # 3

frac_tree_graph_true = np.divide(np.array(n_trees_true_padded[:p+1], dtype=float), n_chordal_true[:p+1])
frac_tree_graph_est = np.divide(n_trees_est, n_chordal_est)

frac_tree_graph_est_means = frac_tree_graph_est.mean(axis=0)
frac_tree_graph_est_sterr = frac_tree_graph_est.std(axis=0)

table_tree.append(frac_tree_graph_true[:p+1]) # 4
table_tree.append(frac_tree_graph_est_means[:p+1]) # 5
table_tree.append(frac_tree_graph_est_sterr[:p+1]) # 6

print("Number of junction trees")
for row in range(1, p+1):
    print str(row) + "\t&\t" + \
          str('%.2E' % Decimal(table_tree[1][row])) + "\t&\t" + \
          str('%.2E' % Decimal(table_tree[2][row])) + "\t&\t" + \
          str('%.2E' % Decimal(table_tree[3][row])) + "\t&\t$\\times 10^{}$" + \
          " \\\\\n" 
          #str('%.2E' % Decimal(table_tree[4][row])) + " \\\\\n"

print("Number of junction trees per chordal graph")
for row in range(1, p+1):
    print str(row) + "\t&\t" + \
          str('%.2E' % Decimal(table_tree[4][row])) + "\t&\t" + \
          str('%.2E' % Decimal(table_tree[5][row])) + "\t&\t" + \
          str('%.2E' % Decimal(table_tree[6][row])) + " &$\\times 10^{}$\\\\\n"

plt.clf()
plt.semilogy(range(1,p+1), frac_tree_graph_est_means[1:], ".-")

# plt.savefig("tree_graph_frac.eps",format="eps")

plt.savefig("frac_tree_graph_p"+str(p)+"_N_"+str(N)+"_T_"+str(T)+".eps", format="eps")