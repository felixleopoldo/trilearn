
"""
A class for handling Markov chains produced from e.g. MCMC.
"""
import json

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph
import pandas as pd

import trilearn.graph.empirical_graph_distribution as gdist
from trilearn.distributions import sequential_junction_tree_distributions as sd


class Trajectory:
    """
    Class for handling trajectories of decomposable graphical models.
    """
    def __init__(self):
        self.trajectory = []
        self.time = []
        self.seqdist = None
        self.burnin = 0
        self.logl = None

    def set_sequential_distribution(self, seqdist):
        """ Set the SequentialJTDistribution for the graphs in the trajectory

        Args:
            seqdist (SequentialJTDistribution): A sequential distribution
        """
        self.seqdist = seqdist

    def set_trajectory(self, trajectory):
        """ Set the trajectory of graphs.

        Args:
            trajectory (Trajectory): An MCMC trajectory of graphs.
        """
        self.trajectory = trajectory

    def set_time(self, generation_time):
        self.time = generation_time

    def add_sample(self, graph, time):
        """ Add graph to the trajectory.

        Args:
            graph (NetworkX graph):
            time (list): List of times it took to generate each sample
        """
        self.trajectory.append(graph)
        self.time.append(time)

    def heatmap(self, from_index=0):
        """ Returns a heatmap of the adjancency matrices in the trajectory.
        """
        length = len(self.trajectory) - from_index
        return np.sum(nx.to_numpy_matrix(g)
                      for g in self.trajectory[from_index:]) / length

    def empirical_distribution(self, from_index=0):
        length = len(self.trajectory) - from_index
        graph_dist = gdist.GraphDistribution()
        for g in self.trajectory[from_index:]:
            graph_dist.add_graph(g, 1./length)
        return graph_dist

    def log_likelihood(self, from_index=0):
        if self.logl is None:
            self.logl = [self.seqdist.ll(g) for g in self.trajectory[from_index:]]
        return pd.Series(self.logl)

    def size(self, from_index=0):
        """ Plots the auto-correlation function of the graph size (number of edges)
        Args:
            from_index (int): Burn-in period, default=0.
        """
        size = [g.size() for g in self.trajectory[from_index:]]
        return pd.Series(size)
        #with sns.axes_style("white"):
        #    autocorrelation_plot(size)

    def write_file(self, filename, optional={}):
        """ Writes a MCMC Trajectory together with the corresponding
        sequential distribution to a json-file.
        """
        js_graphs = [json_graph.node_link_data(graph) for
                     graph in self.trajectory]
        mcmc_traj = {"model": self.seqdist.get_json_model(),
                     "run_time": self.time,
                     "optional": optional,
                     "trajectory": js_graphs}
        with open(filename, 'w') as outfile:
            json.dump(mcmc_traj, outfile)

    def to_json(self, optional={}):
        js_graphs = [json_graph.node_link_data(graph) for
                     graph in self.trajectory]
        mcmc_traj = {"model": self.seqdist.get_json_model(),
                     "run_time": self.time,
                     "optional": optional,
                     "trajectory": js_graphs}
        return mcmc_traj

    def save_to_db(self, db, optional={}):
        js_graphs = [json_graph.node_link_data(graph) for
                     graph in self.trajectory]
        mcmc_traj = {"model": self.seqdist.get_json_model(),
                     "run_time": self.time,
                     "optional": optional,
                     "trajectory": js_graphs}
        db.insert_one(mcmc_traj)

    def from_json(self, mcmc_json):
        graphs = [json_graph.node_link_graph(js_graph)
                  for js_graph in mcmc_json["trajectory"]]
        self.set_trajectory(graphs)
        self.set_time(mcmc_json["run_time"])
        self.optional = mcmc_json["optional"]
        if mcmc_json["model"]["name"] == "ggm_jt_post":
            self.seqdist = sd.GGMJTPosterior()
        self.seqdist.init_model_from_json(mcmc_json["model"])

    def read_file(self, filename):
        """ Reads a trajectory from json-file.
        """
        with open(filename) as mcmc_file:
            mcmc_json = json.load(mcmc_file)

        print mcmc_json.keys()
        self.from_json(mcmc_json)

    def __str__(self):
        ret = "times: " + str(self.time) + \
              "\ngraphs: " + str(self.trajectory) + \
              "\nsequential distribution: " + str(self.seqdist)
        if self.optional:
            ret += "\noptional: " + str(self.optional)
        return ret
