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
        for i in range(0, self.nodes):
            self.pre_capital[i] = self.preferences[i] * self.graph.bonding_capital(i) + (1 - self.preferences[i]) * bridging_capital[i]
            self.post_capital[i] = 0.0

    def observe(self):
        """Returns the current observations that the agents can make
                 of the environment, if applicable.
        """
        observations = []

        for i in range(0, self.nodes):
            # agent id 
            ego = i
            ego_network = self.graph.two_level_ego_network(i)
            # set the first entry of adj matrix as 'ego', to fast locate the ego of the observation
            nodes_order = list(ego_network.nodes())
            nodes_order.remove(ego)
            nodes_order.insert(0, ego)
            adj = nx.adjacency_matrix(ego_network, nodes_order)
            # store adj matrix in dense matrix
            adj = adj.todense()
            # generate distance information
            dist = []
            for j in range(0, len(nodes_order)):
                if adj[0][j] == 0:
                    dist.append(2)
                else:
                    dist.append(1)
            dist[0] = 0
            # generate preference information
            pref = []
            for j in nodes_order:
                pref.append(self.preferences[j])
            # convert adj matrix to a 1xNxN tensor
            adj = torch.from_numpy(np.expand_dims(adj.astype(int), axis=0))
            adj = adj.type(torch.FloatTensor)
            # assemble info matrix. column 1: preferences; column 2: distances
            info = torch.zeros(1, self.nodes, 2, dtype=torch.float)
            for j in range(0, len(nodes_order)):
                info[0,j,0] = pref[j]
                info[0,j,1] = dist[j]
            # assemble final observartions for all agents
            observations.append((adj, nodes_order, info))

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

        for i in range(0, self.nodes):
            self.post_capital[i] = self.preferences[i] * self.graph.bonding_capital(i) + (1 - self.preferences[i]) * bridging_capital[i]
            r = self.post_capital[i] - self.pre_capital[i]
            rewards.append(r) 
        
        return rewards

    def update(self, actions):
        """Update the graph and correspodning information after all agents propose an action
        """
        for i in range(0, self.nodes):
            self.graph.add_edge(i, actions[i])