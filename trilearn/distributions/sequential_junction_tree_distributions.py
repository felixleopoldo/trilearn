"""
Junction tree distributions suitable for SMC sampling.
"""

import numpy as np

import trilearn.graph.decomposable
import trilearn.graph.junction_tree
import trilearn.graph.junction_tree as jtlib
from trilearn.distributions import gaussian_graphical_model
from trilearn.distributions import discrete_dec_log_linear as loglin


class SequentialJTDistribution(object):
    """
    Abstract class of junction tree distributions for SMC sampling.
    """

    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        pass

    def __str__(self):
        pass


class UniformJTDistribution(SequentialJTDistribution):
    """ A sequential formulation of P(T) = P(T|G)P(G), where
        P(G)=1/(#decomopsable graphs)
        and
        P(T|G) = 1/(#junction trees for G).
    """
    def __init__(self, p):
        self.p = p

    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        return 0.0


class CondUniformJTDistribution(SequentialJTDistribution):
    """ A sequential formulation of P(T) = P(T, G) = P(T|G)P(G), where
        P(G)=1/(#decomopsable graphs)
        and
        P(T|G) = 1/(#junction trees for G).
    """
    def __init__(self, p):
        self.p = p

    def ll(self, graph):
        pass

    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        return -trilearn.graph.junction_tree.n_junction_trees_update_ratio(new_separators,
                                                                           old_JT, new_JT)


class LogLinearJTPosterior(SequentialJTDistribution):
    """
    Posterior for a log-linear model.
    """

    def __init__(self, X, cell_alpha, levels, cache_complete_set_prob={},
                 counts={}):
        """
        Args:
            cell_alpha: the constant number of pseudo counts for each cell
            in the full distribution.
        """
        self.p = len(levels)
        self.levels = levels
        self.cache_complete_set_prob = cache_complete_set_prob
        self.cell_alpha = cell_alpha
        self.data = X
        self.no_levels = np.array([len(l) for l in levels])
        self.counts = counts

    def get_json_model(self):
        return {"name": self.__str__(),
                "parameters": {"cell_alpha": self.cell_alpha,
                               "levels": [list(l) for l in self.levels]},
                "data": self.data.tolist()}

    def log_likelihood(self, graph):
        tree = trilearn.graph.decomposable.junction_tree(graph)
        separators = jtlib.separators(tree)
        return loglin.log_likelihood_partial(tree.nodes(), separators, self.no_levels, self.cell_alpha,
                                             self.counts, self.data, self.levels, self.cache_complete_set_prob)

    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        log_mu_ratio = trilearn.graph.junction_tree.n_junction_trees_update_ratio(new_separators,
                                                                                  old_JT, new_JT)
        ll_ratio = self.log_likelihood_diff(old_cliques,
                                            old_separators,
                                            new_cliques,
                                            new_separators,
                                            old_JT,
                                            new_JT)
        return ll_ratio - log_mu_ratio

    def log_likelihood_diff(self, old_cliques, old_separators,
                            new_cliques, new_separators, old_JT, new_JT):
        """ Log-likelihood difference when cliques and separators are added and
            removed.
        """
        old = loglin.log_likelihood_partial(old_cliques, old_separators, self.no_levels, self.cell_alpha,
                                            self.counts, self.data, self.levels, self.cache_complete_set_prob)
        new = loglin.log_likelihood_partial(new_cliques, new_separators, self.no_levels, self.cell_alpha,
                                            self.counts, self.data, self.levels, self.cache_complete_set_prob)
        return new - old

    def __str__(self):
        return "loglin_posterior_n_"+str(self.data.shape[1])+"_p_"+str(self.p)+"_pseudo_obs_"+str(self.cell_alpha)


class GGMJTPosterior(SequentialJTDistribution):
    """ Posterior of Junction tree for a GGM.
    """
    def init_model(self, X, D, delta, cache={}):
        self.parameters = {"delta": delta,
                           "D": D}
        self.SS = X.T * X
        self.X = X
        self.cache = cache

        self.n = X.shape[0]
        self.p = X.shape[1]
        self.idmatrices = [np.identity(i) for i in range(self.p+1)]

    def init_model_from_json(self, sd_json):
        self.init_model(np.asmatrix(sd_json["data"]),
                        np.asmatrix(sd_json["parameters"]["D"]),
                        sd_json["parameters"]["delta"],
                        {})

    def get_json_model(self):

        return {"name": "ggm_jt_post",
                "parameters": {"delta": self.parameters["delta"],
                               "D": self.parameters["D"].tolist()},
                "data": self.X.tolist()}

    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        log_mu_ratio = trilearn.graph.junction_tree.n_junction_trees_update_ratio(new_separators,
                                                                                  old_JT, new_JT)
        log_J_ratio = self.ll_diff(old_cliques,
                                   old_separators,
                                   new_cliques,
                                   new_separators,
                                   old_JT,
                                   new_JT)
        return log_J_ratio - log_mu_ratio

    def ll_diff(self,
                old_cliques,
                old_separators,
                new_cliques,
                new_separators,
                old_JT,
                new_JT):
        old = gaussian_graphical_model.log_likelihood_partial(self.SS, self.n,
                                                              self.parameters["D"],
                                                              self.parameters["delta"],
                                                              old_cliques,
                                                              old_separators,
                                                              self.cache,
                                                              self.idmatrices)

        new = gaussian_graphical_model.log_likelihood_partial(self.SS, self.n,
                                                              self.parameters["D"],
                                                              self.parameters["delta"],
                                                              new_cliques,
                                                              new_separators,
                                                              self.cache,
                                                              self.idmatrices)

        return new - old

    def log_likelihood(self, graph):
        return gaussian_graphical_model.log_likelihood(graph, self.SS, self.n,
                                                       self.parameters["D"],
                                                       self.parameters["delta"],
                                                       self.cache)

    def log_likelihood_partial(self, cliques, separators):
        return gaussian_graphical_model.log_likelihood_partial(self.SS, self.n,
                                                               self.parameters["D"],
                                                               self.parameters["delta"],
                                                               cliques, separators, self.cache)

    def __str__(self):
        return "ggm_posterior_n_" + str(self.n) + "_p_" + str(self.p) + "_prior_scale_" + str(
            self.parameters["delta"]) + "_shape_x"
