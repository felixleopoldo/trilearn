import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import trilearn.graph.graph as glib
import trilearn.graph.junction_tree as jtlib
from trilearn.auxiliary_functions import sample_classification_datasets
from trilearn.distributions import g_intra_class
from trilearn.graph_predictive import GraphPredictive

np.random.seed(1)

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

df_full = sample_classification_datasets(mus, covmats, n_samples_in_each_class=n_samples)

df_full.to_csv("sample_data/classification_full_dataset.csv", index=False)
df_full = pd.read_csv("sample_data/classification_full_dataset.csv")

x_train, x_test, y_train, y_test = train_test_split(df_full.drop(["y"], axis=1), df_full["y"],
                                                    test_size=0.3, random_state=1)
# Comparison
for name, clf in zip(names, classifiers):
    clf.fit(x_train.get_values(), y_train.get_values())
    print str(name) + " " + str(clf.score(x_test.get_values(),
                                          y_test.get_values()))

