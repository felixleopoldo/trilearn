import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import trilearn.graph.junction_tree as jtlib
import trilearn.graph.graph as glib
from trilearn.distributions import g_intra_class
import trilearn.smc as smc
import trilearn.graph_predictive as gpred

np.random.seed(1)

def sample_classification_datasets(mus, covmats, n_samples):
    # Generate training data
    n_train = [n_samples] * n_classes
    x = np.matrix(np.zeros((sum(n_train), n_dim))).reshape(sum(n_train), n_dim)
    y = np.matrix(np.zeros((sum(n_train), 1), dtype=int)).reshape(sum(n_train), 1)

    for c in range(n_classes):
        fr = sum(n_train[:c])
        to = sum(n_train[:c + 1])
        x[np.ix_(range(fr, to),
                       range(n_dim))] = np.matrix(np.random.multivariate_normal(
                                                  np.array(mus[c]).flatten(),
                                                  covmats[c],
                                                  n_train[c]))
        y[np.ix_(range(fr, to), [0])] = np.matrix(np.ones(n_train[c], dtype=int) * c).T

    return x, y

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "LDA", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
    QuadraticDiscriminantAnalysis()]


n_classes = 2
classes = range(n_classes)
n_dim = 4

# Sample graphs
graphs = [jtlib.sample(n_dim).to_graph() for _ in classes]
for g in graphs:
    glib.plot_adjmat(g)
    #plt.show()

# Generate covariance matrices
sigma2 = 1.0
rho = 0.5
covmats = [g_intra_class.cov_matrix(graphs[c], rho, sigma2) for c in classes]

# Set means
mu_shift = 0.001
mus = [np.matrix(np.ones(n_dim)).T * i * mu_shift for i in classes]

# Sample class data
n_samples = n_dim + 1
x_train, y_train = sample_classification_datasets(mus, covmats, n_samples)
x_test, y_test = sample_classification_datasets(mus, covmats, n_samples)
data = np.column_stack((y_train, x_train))
ys = pd.Series(np.array(y_train).flatten(), dtype=int)
df = pd.DataFrame(x_train)
df["y"] = ys

df = df[["y"] +  range(n_dim)]
df.to_csv("classification_dataset.csv", header=False, index=False)

df = pd.read_csv("classification_dataset.csv", header=None)

# Comparison with other methods
for name, clf in zip(names, classifiers):
    clf.fit(np.array(x_train), np.array(y_train).flatten())
    print str(name) + " " + str(clf.score(np.array(x_test), np.array(y_test).flatten()))

# Generate MCMC chains

# trajs = [None] * n_classes
# for c in classes:
#     x = x_train[y==c]
#     trajs[c] = smc.gen_pgibbs_ggm_trajectories(n_particles=30, n_pgibbs_samples=300, async=False, centered=False)
#
# graph_distributions = [None] * n_classes
# for c in classes:
#     graph_distributions[c] = trajs[c].empirical_distribution()
# predclass.fit(x_train, y_train, graph_distributions)

predclass = gpred.GraphPredictive(x_train, y_train)
print "Predictive: " +str(predclass.score(x_test, y_test))
predclass.gen_gibbs_chains(n_particles=30, n_pgibbs_samples=30, async=False)
predclass.set_graph_dists(set_burnins=True)
# predclass.plot_class_heatmap(0)
# plt.show()
# predclass.plot_class_heatmap(1)
# plt.show()
#
print "Graphical predictive: " +str(predclass.score(x_test, y_test))
#
