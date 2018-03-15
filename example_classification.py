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

from sklearn.model_selection import train_test_split

import trilearn.graph.junction_tree as jtlib
import trilearn.graph.graph as glib
from trilearn.distributions import g_intra_class
import trilearn.smc as smc
from trilearn.graph_predictive import GraphPredictive

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
         "Naive Bayes", "LDA", "QDA", "BayesPred", "BayesGraphPred"]

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
    QuadraticDiscriminantAnalysis(),
    GraphPredictive(standard_bayes=True),
    GraphPredictive(n_particles=100, n_pgibbs_samples=5)]


n_classes = 2
classes = range(n_classes)
n_dim = 20

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
n_samples = n_dim + 50

x_full, y_full = sample_classification_datasets(mus, covmats, n_samples)
x_train, y_train = sample_classification_datasets(mus, covmats, n_samples)
x_test, y_test = sample_classification_datasets(mus, covmats, n_samples)


ys_full = pd.Series(np.array(y_train).flatten(), dtype=int)
df_full = pd.DataFrame(x_train)
df_full["y"] = ys_full
df_full = df_full[["y"] +  range(n_dim)]
df_full.columns = ["y"] + ["x" + str(i) for i in range(1, n_dim+1)]
df_full.to_csv("classification_full_dataset.csv", index=False)

ys_train = pd.Series(np.array(y_train).flatten(), dtype=int)
df_train = pd.DataFrame(x_train)
df_train["y"] = ys_train
df_train = df_train[["y"] +  range(n_dim)]
df_train.columns = ["y"] + ["x" + str(i) for i in range(1, n_dim+1)]
df_train.to_csv("classification_train_dataset.csv", index=False)

ys_test = pd.Series(np.array(y_test).flatten(), dtype=int)
df_test = pd.DataFrame(x_test)
df_test["y"] = ys_test
df_test = df_test[["y"] +  range(n_dim)]
df_test.columns = ["y"] + ["x" + str(i) for i in range(1, n_dim+1)]
df_test.to_csv("classification_test_dataset.csv", index=False)

df_full = pd.read_csv("classification_full_dataset.csv")
df_train = pd.read_csv("classification_train_dataset.csv")
df_test = pd.read_csv("classification_test_dataset.csv")

y_train = df_train["y"]
x_train = df_train.drop(["y"], axis=1)
y_test = df_test["y"]
x_test = df_test.drop(["y"], axis=1)

x_train, x_test, y_train, y_test = train_test_split(df_full.drop(["y"], axis=1), df_full["y"],
                                                    test_size=0.3, random_state=1)

# Comparison
for name, clf in zip(names, classifiers):
    clf.fit(x_train.get_values(), y_train.get_values())
    print str(name) + " " + str(clf.score(x_test.get_values(),
                                          y_test.get_values()))

