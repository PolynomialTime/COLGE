import numpy as np
import torch
#import pulp
import graph 
import runner
import networkx as nx
import copy
import multiprocessing
#import community
from networkx.algorithms.community import greedy_modularity_communities

multiprocessing.set_start_method('spawn', True)
"""
This file contains the definition of the environment
in which the agents are run.
"""


class Environment:
    def __init__(self, graph_init:graph.Graph, env_class):
        self.device = torch.device('cpu')
        self.graph_init = graph_init
        self.preferences = []
        if env_class == 'community':
            self.preferences = [1.0 for i in range(self.graph_init.nodes_nbr())]
        if env_class == 'small-world':
            self.preferences = [0.0 for i in range(self.graph_init.nodes_nbr())]


    def reset(self):
        """Reset the enviornment at the beginning of an episode
        """
        self.graph = copy.deepcopy(self.graph_init)
        self.nodes = self.graph.nodes()
        self.nodes_nbr = self.graph.nodes_nbr()
        self.pre_capital = [0.0 for i in range(self.nodes_nbr)]
        self.post_capital = [0.0 for i in range(self.nodes_nbr)]
        self.iter = 0

        # rewards of last step
        bridging_capital = self.graph.bridging_capital()
        #print(self.nodes)
        for i in self.nodes:
            self.pre_capital[i] = self.preferences[i] * self.graph.bonding_capital(i) + (1 - self.preferences[i]) * bridging_capital[i]
            self.post_capital[i] = 0.0

    """
    Obsrvation format: ( 0: adjacency matrix, 1: ordered nodes, 2: (preferecens, distances), 3: iteration )

    """

    def observe(self):
        """Returns the current observations that the agents can make
                 of the environment, if applicable.
        """
        observations = []
        results = []
        pool = multiprocessing.Pool(processes=6)
        for i in self.graph.nodes():
            results.append(pool.apply_async(self.get_observation, (i,)))
        pool.close() # close pool
        pool.join() # wait for all jobs in the pool finished
        
        for res in results:
            observations.append(res.get())
        return observations

    
    def get_observation(self, i):
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
        for j in range(len(nodes_order)):
            if adj[0, j] == 0:
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
        adj.to(self.device)
        # assemble info matrix. column 1: preferences; column 2: distances
        '''
        info = torch.zeros(1, self.nodes_nbr, 2, dtype=torch.float)
        for j in range(0, len(nodes_order)):
            info[0,j,0] = pref[j]
            info[0,j,1] = dist[j]
        '''
        info = torch.zeros(1, len(nodes_order), 1, dtype=torch.float).to(self.device)
        for j in range(len(nodes_order)):
            info[0,j,0] = dist[j]
        # assemble final observartions for all agents
        return (adj, nodes_order, info, self.iter)



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
        bonding_capital = []
        bridging_capital = self.graph.bridging_capital()

        pool = multiprocessing.Pool(processes=6)
        results = []
        for i in self.graph.nodes():
            results.append(pool.apply_async(self.graph.bonding_capital, (i,)))
        pool.close() # close pool
        pool.join() # wait for all jobs in the pool finished
        for res in results:
            bonding_capital.append(res.get())

        for i in iter(self.graph.nodes()):
            self.post_capital[i] = self.preferences[i] * bonding_capital[i] + (1 - self.preferences[i]) * bridging_capital[i]
            r = self.post_capital[i] - self.pre_capital[i]
            rewards.append(r)
        
        return rewards

    def update(self, actions):
        """Update the graph and correspodning information after all agents propose an action
        """
        for i in self.nodes:
            self.graph.add_edge(i, actions[i])
        
        self.iter += 1

    def community_info(self):
        #part = list(community.best_partition(self.graph.g))
        c = list(greedy_modularity_communities(self.graph.g))
        a = nx.algorithms.community.modularity(self.graph.g, c)
        #b = nx.algorithms.community.modularity(self.graph.g, part)
        return a

    def small_world_info(self):
        return (nx.average_shortest_path_length(self.graph.g), nx.average_clustering(self.graph.g))