""" A Bayesian graphical predictive classifier.
"""
from multiprocessing import Pool
import itertools

import numpy as np
import scipy
import networkx as nx
import matplotlib.pyplot as plt

import trilearn.graph.trajectory as mc
import trilearn.graph.graph as glib
import trilearn.graph.decomposable as dlib
import trilearn.graph.junction_tree as jtlib
import trilearn.graph.empirical_graph_distribution as gdist
import trilearn.smc as smc

from trilearn.distributions import multivariate_students_t as tdist
from trilearn.distributions import sequential_junction_tree_distributions as seqjtdist


scipy.set_printoptions(precision=2, suppress=True)

class GraphPredictive:
    def __init__(self,
                 n_particles=None, n_pgibbs_samples=None,
                 prompt_burnin=False,
                 standard_bayes=False,
                 async=False):
        self.n_particles = n_particles
        self.n_pgibbs_samples = n_pgibbs_samples
        self.prompt_burnin = prompt_burnin
        self.standard_bayes = standard_bayes
        self.async = async

    def fit(self, x, y, hyper_mu=None, hyper_v=None, hyper_tau=None, hyper_alpha=None,
            same_graph_groups=None):
        """ These parameters are set here in the constructor in order to avoid
        mismatch since the hyper parameters in classification has to be
        consistent with those in the structure learning procedure.

        Args:
            x (Numpy matrix): Matrix of training data
            y (Numpy array): Array of class correspondence
            hyper_mu (Numpy array): Array of mean hyper parameter for
            the normal inverse Wishart density
            hyper_v (float): Parameter in the covariace matrix in the normal inverse Wishart density
            hyper_tau (Numpy matrix): Precision matrix in the normal inverse Wishart density
            hyper_alpha (float): Degrees of freedom in the normal inverse Wishart density
        """
        self.classes = list(set(np.array(y).flatten()))
        classes   = self.classes
        self.x = np.matrix(x)
        self.y = np.matrix(y)
        self.p = x.shape[1]

#        classes = list(set(np.array(y).flatten()))
        n_classes = len(self.classes)
        n_dim = x.shape[1]

        if same_graph_groups is None:
            self.same_graph_groups = [[i] for i in classes]
        else:
            self.same_graph_groups = same_graph_groups

        n_groups = len(self.same_graph_groups)
        if hyper_mu is None:
            mu_shift = 0.001
            self.hyper_mu = [np.matrix(np.ones(n_dim)).T * i * mu_shift for i in classes]
        else:
            self.hyper_mu = hyper_mu
        if hyper_tau is None:
            self.hyper_tau = [np.matrix(np.identity(n_dim))] * n_groups  # needs to be small
        else:
            self.hyper_tau = hyper_tau
        if hyper_v is None:
            self.hyper_v = [1.] * n_classes
        else:
            self.hyper_v = hyper_v
        if hyper_alpha is None:
            self.hyper_alpha = [n_dim + 1] * n_groups
        else:
            self.hyper_alpha = hyper_alpha

        self.graph_dists = None
        self.ggm_trajs = None

        if self.standard_bayes is True:
            # Initiate to the complete graph, corresponding to the
            # standard Bayesian predictive classifier
            graph = nx.complete_graph(n_dim)
            self.graph_dists = [None for _ in self.same_graph_groups]
            for g, group in enumerate(self.same_graph_groups):
                self.graph_dists[g] = gdist.GraphDistribution()
                self.graph_dists[g].add_graph(graph, 1.0)

        else:
            self.gen_gibbs_chains(n_particles=self.n_particles,
                                  n_pgibbs_samples=self.n_pgibbs_samples,
                                  async=False)  # move to fit
            self.set_graph_dists(set_burnins=self.prompt_burnin)  # move to fit

    # def fit(self, x, y):
    #     """
    #     Sets the training data with class belongings.
    #
    #     Args:
    #         x (Numpy matrix): Matrix of training data
    #         y (Numpy array): Array of class correspondence. Eg. [0,0,1]
    #     indicates that x[0] and x[1] belongs to class 0 and x[2] belongs to
    #     class 1.
    #
    #     """

    def set_hyper_parameters(self,
                             hyper_mu,
                             hyper_v,
                             hyper_tau,
                             hyper_alpha):
        """
        Args:
            hyper_mu (Numpy array): Array of mean hyper parameter for the normal inverse wishart density
            hyper_v (float): Parameter in the covariance matrix in the normal inverse wishart density
            hyper_tau (Numpy matrix): Precision matrix in the normal inverse wishart density
            hyper_alpha (float): Degrees of freedom in the normal inverse wishart density
        """

        self.hyper_mu = hyper_mu
        self.hyper_v = hyper_v
        self.hyper_alpha = hyper_alpha
        self.hyper_tau = hyper_tau

    def gen_gibbs_chains(self, n_particles, n_pgibbs_samples, smc_radius=None, cta_alpha=0.5, cta_beta=0.5, async=True):
        """ If same_graph is True, this generates one single Gibbs graph-trajectory
        for common for all classes.
        Otherwise, this generates Gibbs graph-trajectories for each class.

        Args:
            smc_radius: radius for the SMC algorithm.
        """
        self.smc_N = n_particles
        if smc_radius is None:
            self.smc_radius = self.x.shape[1]
        else:
            self.smc_radius = smc_radius
        self.pgibbs_T = n_pgibbs_samples
        self.cta_alpha = cta_alpha
        self.cta_beta = cta_beta
        self.ggm_trajs = [None for _ in self.same_graph_groups]

        pool = None
        async_results = None
        if async is True:
            pool = Pool(processes=len(self.classes))

        async_results = [None for _ in self.same_graph_groups]
        for g, group in enumerate(self.same_graph_groups):
            x_centered = np.array([])
            # concatenate the data, centered by mean in each class
            for c in group:
                c_inds = (np.array(self.y).ravel() == c)
                xc = self.x[np.ix_(c_inds, range(self.p))]
                xc_centered = xc - np.mean(xc, axis=0)
                if len(x_centered) == 0:
                    x_centered = xc_centered
                else:
                    x_centered = np.concatenate((x_centered, xc_centered),
                                                axis=0)
            seq_dist = seqjtdist.GGMJTPosterior()
            cache = {}
            seq_dist.init_model(x_centered, self.hyper_tau[g],
                                self.hyper_alpha[g], cache)
            if async is True:
                async_results[g] = pool.apply_async(smc.particle_gibbs,
                                                    (n_particles, cta_alpha,
                                                     cta_beta,
                                                     self.smc_radius, n_pgibbs_samples,
                                                     seq_dist))
            else:
                self.ggm_trajs[g] = smc.particle_gibbs(n_particles, cta_alpha,
                                                       cta_beta,
                                                       self.smc_radius,
                                                       n_pgibbs_samples,
                                                       seq_dist)
        if async is True:
            for g in range(len(self.same_graph_groups)):
                self.ggm_trajs[g] = async_results[g].get()

    def gibbs_chains_to_json(self, title, optional={}):
        """ Returns the Gibbs trajectory in json format.

        Args:
            title (string): The json key for the attriute _id
            optional (dict): Optional infor in json format to be included in the json object.
        """
        cln = {"_id": title}
        cln["optional"] = optional
        cln["x"] = self.x.tolist()
        cln["y"] = self.y.tolist()
        cln["smc_setting"] = {"cta": {"alpha": self.cta_alpha,
                                      "beta": self.cta_beta},
                              "pgibbs": {"trajectory_lengths": self.pgibbs_T,
                                         "particles": self.smc_N,
                                         "delta": self.p}}

        cln["hyper_params"] = {"tau": [tmp.tolist() for tmp in self.hyper_tau],
                               "alpha": self.hyper_alpha,
                               "mu": [tmp.tolist() for tmp in self.hyper_mu],
                               "v": self.hyper_v}

        cln["same_graph_groups"] = self.same_graph_groups
        cln["markov_chains"] = [None for _ in self.same_graph_groups]
        if self.ggm_trajs is not None:
            for g in range(len(self.same_graph_groups)):
                cln["markov_chains"][g] = self.ggm_trajs[g].to_json()

        else:
            print "No trajectories"

        return cln

    def gibbs_chains_from_json(self, gibbs_js):
        """ Reads a Gibbs trajectory in json format.

        Args:
            gibbs_js (dict): Gibbs trajectory in json format.

        """
        if self.same_graph is True:
            self.graph_dist = self.ggm_traj.empirical_distribution(self.burnin)
        else:
            self.graph_dists = [None for _ in self.same_graph_groups]
            for g in range(len(self.same_graph_groups)):
                self.graph_dists[g] = self.ggm_trajs[g].empirical_distribution(self.burnins[g])

    def set_burnin(self, true_graphs=None, directory=".", title=""):
        """ Sets the burn-in period for the class-groups.
        """
        print("Look at the plot, close it, and set the burn-in")
        self.burnins = [None for _ in self.same_graph_groups]
        saturated_model = nx.complete_graph(self.p)
        for c in range(len(self.same_graph_groups)):
            plt.clf()
            plt.title('Graph log-likelihood. Class group ' + str(c) + '.')
            self.ggm_trajs[c].likelihood().plot()
            chain_length = len(self.ggm_trajs[c].trajectory)
            #plt.plot([self.ggm_trajs[c].seqdist.ll(saturated_model)] *
            #         chain_length, 'red')
            if true_graphs is not None:
                plt.plot([self.ggm_trajs[c].seqdist.ll(true_graphs[c])] *
                         chain_length, 'green')
            # fig = plt.gcf()
            # fig.savefig(directory +
            #             "/"+title+"_ll_class-group_"+str(c)+".eps",
            #             format="eps",
            #             bbox_inches='tight', dpi=300)
            plt.show()
            try:
                self.burnins[c] = int(raw_input("Burn-in for class group "+str(c)+" (default=0): "))
            except ValueError:
                self.burnins[c] = 0

    def graph_dists_to_json(self, dists_ids, optional={}):
        json_dists = []
        if self.graph_dists is None:
            print "You need to run set_graph_dists()"
        else:
            for c in range(len(self.same_graph_groups)):
                optional["burnin-in"] = self.burnins[c]  # BUG ?
                optional["class"] = c
                # set same graph groups
                json_dists.append(self.graph_dists[c].to_json(c,
                                                              optional=optional))
        return json_dists

    def graph_dists_from_json(self, json_graph_dists):
        self.graph_dists = [gdist.GraphDistribution() for _ in self.same_graph_groups]
        for c in range(len(self.same_graph_groups)):
            self.graph_dists[c].from_json(json_graph_dists[c])

    def get_group(self, c):
        for g, g_list in enumerate(self.same_graph_groups):
            if c in g_list:
                return g

    def plot_class_heatmap(self, c):
        group = self.get_group(c)
        self.plot_group_heatmap(group)

    def plot_group_heatmap(self, group):
        self.ggm_trajs[group].plot_heatmap(self.burnins[group])
        plt.title('Edge heatmap. Class group: ' + str(group) + ', burn-in: ' + str(self.burnins[group])+'.')
        plt.yticks(rotation=0)

    def set_graph_dists(self, true_graphs=None, json_trajs=None,
                        directory=".", title="", set_burnins=False):

        self.graph_dists = [None for _ in self.same_graph_groups]
        self.burnins = [0 for _ in self.same_graph_groups]
        if json_trajs is None:
            if set_burnins:
                self.set_burnin(true_graphs=true_graphs, directory=directory,
                                title=title)
            for c in range(len(self.same_graph_groups)):
                self.graph_dists[c] = self.ggm_trajs[c].empirical_distribution(self.burnins[c])
        else:
            self.ggm_trajs = [None for _ in self.classes]
            for c in range(len(self.same_graph_groups)):
                self.ggm_trajs[c] = mc.MCMCTrajectory()
                self.ggm_trajs[c].from_json(json_trajs[c])
            if set_burnins:
                self.set_burnin(true_graphs=true_graphs, directory=directory,
                                title=title)
            for c in range(len(self.same_graph_groups)):
                # plt.clf()
                # self.ggm_trajs[c].plot_heatmap(self.burnins[c])
                # plt.title('Edge heatmap')
                # plt.yticks(rotation=0)
                # plt.savefig(directory + "/" + title +
                #             "_edge_heatmap_burnin_"+str(self.burnins[c]) +
                #             "_class_"+str(c)+".eps",
                #             format="eps", bbox_inches='tight', dpi=300)
                # self.graph_dists[c] = self.ggm_trajs[c].empirical_distribution(self.burnins[c])
                # plt.clf()
                # self.ggm_trajs[c].plot_autocorr(self.burnins[c])
                # #plt.title('Auto correlation size')
                # plt.savefig(directory + "/" + title +
                #             "_autocorr_size_burnin_"+str(self.burnins[c]) +
                #             "_class_"+str(c)+".eps",
                #             format="eps", bbox_inches='tight', dpi=300)
                self.graph_dists[c] = self.ggm_trajs[c].empirical_distribution(self.burnins[c])

    def score(self, x_test, y_test):
        pred_y = []
        for i, xval in enumerate(x_test):
            # print "Predicting: " + str(i+1) + "/" + str(len(x_test))
            pred_y.append(self.predict(np.asmatrix(xval)))
        pred_y = np.array(pred_y)
        return np.sum(pred_y == np.array(y_test).flatten())/float(len(x_test))

    def predict(self, x_new):
        """
        Model:
        x_i | M=m, R=r ~ Normal(m,r)
        R ~ Wishart(tau, alpha)
        M | R=r ~ Normal(mu, r*v)
        Args:
        x_new: row matrix with new observations for which we
              the predictive density will be computed.
              Example 2-dim x. [[0.4, 0.5],
        [0.2, 1.4]]
        x: row matrix with traning data
        y: vector of classes
        mu: hyper parameter for M
        v: hyper parameter for M
        classes: unique class labels, eg. [0, 1, 2]
        graph_distribution: dictionary with graph distribution for each class.
        """
        p = self.p
        # graph_dists = None
        # if self.graph_dists is None:
        #     graph = nx.complete_graph(p)
        #     graph_dists = [None for _ in self.same_graph_groups]
        #     for g, group in enumerate(self.same_graph_groups):
        #         graph_dists[g] = gdist.GraphDistribution()
        #         graph_dists[g].add_graph(graph, 1.0)
        # else:
        #     graph_dists = self.graph_dists

        graph_dists = self.graph_dists

        ctg = [0] * len(self.classes)  # class to group
        for gi, g in enumerate(self.same_graph_groups):
            for c in g:
                ctg[c] = gi

        pred_dist = {}
        pred_dist_simult = {}
        class_space = [self.classes] * len(x_new)
        for classification in itertools.product(*class_space):
            pred_dist_simult[classification] = 0.0
            for c in classification:
                if c not in pred_dist:
                    c_inds = (np.array(self.y).ravel() == c)
                    xc = self.x[np.ix_(c_inds, range(p))]

                    pred_dist[c] = self.predictive_pdf(x_new,
                                                       xc,
                                                       self.hyper_mu[ctg[c]],
                                                       self.hyper_v[ctg[c]],
                                                       self.hyper_tau[ctg[c]],
                                                       self.hyper_alpha[ctg[c]],
                                                       graph_dists[ctg[c]])
                    pred_dist_simult[classification] *= pred_dist[c]

        return int(max(pred_dist, key=pred_dist.get))

    def predictive_pdf(self, x_new, x, mu, v, tau, alpha, graph_dist):
        """
        This is the predictive distribution of x_new.
        It is a multivatiate T-distributiona where the graph
        is marginalized out accordning to according to graph_dist.
        """
        pred_density = 0.0

        cache = {}
        for graph in graph_dist.domain:
            log_pred_density = 0.0
            tree = dlib.junction_tree(graph)
            cliques = tree.nodes()
            separators = jtlib.separators(tree)

            k = x_new.shape[0]  # items to classify
            n = x.shape[0]
            mu = np.matrix(mu).reshape(len(mu), 1)
            x_bar = np.mean(x, axis=0).T
            s = (x-mu.T).T * (x-mu.T)
            mu_star = (v*mu + n*x_bar) / (v + n)
            tau_star = tau + s + (n*v / (v + n)) * (mu - x_bar)*(mu - x_bar).T
            v_star = v + n
            for c in cliques:
                if c not in cache:
                    node_list = list(c)
                    x_new_c = x_new[np.ix_(range(k), node_list)].T
                    t_d = len(c)
                    t_mu = mu_star[node_list]
                    t_tau_star = (tau_star[np.ix_(node_list, node_list)] *
                                  (v_star + 1) / (v_star * (v_star + 1 - t_d))).I
                    t_df = v_star - t_d + 1
                    cache[c] = tdist.log_pdf(x_new_c,
                                             t_mu,
                                             t_tau_star,
                                             t_df)
                log_pred_density += cache[c]

            for sep in separators:
                if len(sep) == 0:
                    continue
                nu = len(separators[sep])
                if sep not in cache:
                    node_list = list(sep)

                    x_new_s = x_new[np.ix_(range(k), node_list)].T
                    t_d = len(sep)
                    t_mu = mu_star[node_list]
                    t_tau_star = (tau_star[np.ix_(node_list, node_list)] *
                                  (v_star + 1) / (v_star * (v_star + 1 - t_d))).I
                    t_df = v_star - t_d + 1
                    cache[sep] = tdist.log_pdf(x_new_s,
                                               t_mu,
                                               t_tau_star,
                                               t_df)
                log_pred_density -= nu * cache[sep]
            pred_density += np.exp(log_pred_density) * graph_dist.prob(graph)
        return float(pred_density)
