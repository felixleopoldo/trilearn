from operator import itemgetter

import networkx as nx
import numpy as np
from networkx.readwrite import json_graph

import trilearn.graph.graph as glib


class GraphDistribution(object):
    """
    A discrete graph distribution.
    """
    def __init__(self):
        self.distribution = {}
        self.domain = set({})
        self.optional = None

    def to_json(self, dist_id, optional={}):
        """
        Distribution is list of pairs of json_graphs and corresponding
        probabilities.

        """
        dist_json = {"_id": dist_id,
                     "distribution": [],
                     "optional": optional}
        for key, val in self.distribution.iteritems():
            dist_json["distribution"].append([json_graph.node_link_data(
                val["graph"]),
                val["prob"]])
        return dist_json

    def heatmap(self):
        graphprobs = [(self.distribution[d]["graph"], self.distribution[d]["prob"]) for d in self.distribution]
        p = graphprobs[0][0].order()
        heatmap = np.matrix(np.zeros(p*p).reshape(p, p))
        for graph, prob in graphprobs:
            heatmap += nx.to_numpy_matrix(graph) * prob
        return heatmap

    def from_json(self, js_distribution):
        self.optional = js_distribution["optional"]
        for graph, prob in js_distribution["distribution"]:
            self.add_graph(json_graph.node_link_graph(graph), prob)

    def read_from_dict(self, dict_distr):
        """
        Format: {nx.Graph(): probabilty,...}
        """
        for graph, prob in dict_distr:
            self.add_graph(graph, prob)

    def add_graph(self, graph, prob):
        """
        Adds graph to the distribution. It graph does not exists,then
        graph gets probability prob, otherwise prob is added to the
        esisting probability.
        """
        if glib.hash_graph(graph) not in self.distribution:
            self.distribution[glib.hash_graph(graph)] = {"graph": graph,
                                                         "prob": prob}
            self.domain.add(graph)
        else:
            self.distribution[glib.hash_graph(graph)]["prob"] += prob

    def prob(self, graph):
        return self.distribution[glib.hash_graph(graph)]["prob"]

    def __str__(self):
        tmp = [(val["graph"].edges(), val["prob"]) for
               key, val in self.distribution.iteritems()]
        return str(tmp)

    def mode(self, number=1):
        graphs_probs = [(val["graph"], val["prob"]) for _, val in self.distribution.iteritems()]
        graphs_probs.sort(key=itemgetter(1), reverse=True)
        return graphs_probs[:number]