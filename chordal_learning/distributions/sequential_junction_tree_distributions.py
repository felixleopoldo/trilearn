"""
Junction tree distributions suitable for SMC sampling.
"""

import numpy as np

import chordal_learning.graph.christmas_tree_algorithm as jtexp
import chordal_learning.graph.graph as glib
import chordal_learning.graph.junction_tree as jtlib
from chordal_learning.distributions import gaussian_graphical_model
from chordal_learning.distributions import discrete_dec_log_linear as loglin


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

    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        return -jtexp.mu_update_ratio(new_separators,
                                      old_JT, new_JT)


class LogLinearJTPosterior(SequentialJTDistribution):
    """
    Posterior for a log-linear model.
    """

    def __init__(self, X, cell_alpha, levels, cache, counts={}):
        """
        Args:
            cell_alpha: the constant number of pseudo counts for each cell
            in the full distribution.
        """
        self.p = len(levels)
        self.levels = levels
        self.cache = cache
        self.cell_alpha = cell_alpha
        self.data = X
        self.no_levels = np.array([len(l) for l in levels])
        self.counts = counts

    def get_json_model(self):
        return {"name": self.__str__(),
                "parameters": {"cell_alpha": self.cell_alpha,
                               "levels": [list(l) for l in self.levels]},
                "data": self.data.tolist()}

    def ll(self, graph):
        tree = glib.junction_tree(graph)
        separators = jtlib.separators(tree)
        return loglin.log_likelihood_partial(tree.nodes(), separators, self.no_levels, self.cell_alpha,
                                             self.counts, self.data, self.levels, self.cache)

    def log_ratio(self,
                  old_cliques,
                  old_separators,
                  new_cliques,
                  new_separators,
                  old_JT,
                  new_JT):
        log_mu_ratio = jtexp.mu_update_ratio(new_separators,
                                             old_JT, new_JT)
        ll_ratio = self.ll_diff(old_cliques,
                                old_separators,
                                new_cliques,
                                new_separators,
                                old_JT,
                                new_JT)
        return ll_ratio - log_mu_ratio

    def ll_diff(self, old_cliques, old_separators,
                new_cliques, new_separators, old_JT, new_JT):
        """ Log-likelihood difference when cliques and separators are added and
            removed.
        """
        old = loglin.log_likelihood_partial(old_cliques, old_separators, self.no_levels, self.cell_alpha,
                                            self.counts, self.data, self.levels, self.cache)
        new = loglin.log_likelihood_partial(new_cliques, new_separators, self.no_levels, self.cell_alpha,
                                            self.counts, self.data, self.levels, self.cache)
        return new - old


    def __str__(self):
        return "loglin_pseudo_obs_"+str(self.cell_alpha)


class GGMJTPosterior(SequentialJTDistribution):
    """ Posterior of Junction tree for a GGM.
    """
    def init_model(self, X, D, delta, cache):
        self.parameters = {"delta": delta,
                           "D": D}
        self.SS = X.T * X
        self.X = X
        self.cache = cache
        self.n = X.shape[0]
        self.p = X.shape[1]
        self.idmatrices = [np.identity(i) for i in range(self.p+1)]

    def get_json_model(self):

        return {"name": self.__str__(),
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
        log_mu_ratio = jtexp.mu_update_ratio(new_separators,
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

    def ll(self, graph):
        return gaussian_graphical_model.log_likelihood(graph, self.SS, self.n,
                                                       self.parameters["D"],
                                                       self.parameters["delta"],
                                                       self.cache)

    def __str__(self):
        return "ggm_jt_post"
