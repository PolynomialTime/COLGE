import numpy as np
import torch
import pulp
import graph 
import runner
import networkx as nx
import copy

"""
This file contains the definition of the environment
in which the agents are run.
"""


class Environment:
    def __init__(self, graph_init:graph.Graph, preferences):
        self.graph_init = graph_init
        self.preferences = preferences

    def reset(self):
        """Reset the enviornment at the beginning of an episode
        """
        self.graph = copy.deepcopy(self.graph_init)
        self.nodes = self.graph.nodes()
        self.nbr_of_nodes = len(self.nodes)
        self.edge_add_old = 0
        self.pre_capital = []
        self.post_capital = []
        # rewards of last step
        bridging_capital = self.graph.bridging_capital()
        for i in self.nodes:
            self.pre_capital[i] = self.preferences[i] * self.graph.bonding_capital(i) + (1 - self.preferences[i]) * bridging_capital[i]
            self.post_capital[i] = 0.0

    def observe(self):
        """Returns the current observations that the agents can make
                 of the environment, if applicable.
        """
        observations = []
        for i in self.graph.nodes:
            ego = i # agent id 
            ego_network = self.graph.two_level_ego_network(i)
            ego_network_adj = nx.adjacency_matrix(ego_network).todense() # store in matrix
            ego_network_nodes = ego_network.nodes()
            # observartions
            observations.append((ego, ego_network_adj, ego_network_nodes))

        return observations

    def act(self, actions):
        """Update the environment after all agents propose actions
           Return rewards of agents
        """
        # update graph
        self.update(actions)
        # compute rewards
        rewards = self.get_rewards()
        # update capital information
        self.pre_capital = self.post_capital
        return rewards

    def get_rewards(self):
        # socail capital of all agents after acting
        rewards = []
        bridging_capital = self.graph.bridging_capital()
        for i in self.graph.nodes:
            self.post_capital[i] = self.preferences[i] * self.graph.bonding_capital(i) + (1 - self.preferences[i]) * bridging_capital[i]
            r = self.post_capital[i] - self.pre_capital[i]
            rewards.append(r) 
        return rewards

    def update(self, actions):
        """Update the graph and correspodning information after all agents propose an action
        """
        for i in self.nodes:
            self.graph.add_edge(i, actions[i])