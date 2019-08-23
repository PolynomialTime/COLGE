import torch
import numpy as np
import networkx as nx
import graph


g = graph.Graph('cycle_graph', 100, 1)
ego_network = g.two_level_ego_network(1)
adj = nx.adjacency_matrix(ego_network).todense() # store in matrix
adj = torch.from_numpy(np.expand_dims(adj.astype(int), axis=0))
adj = adj.type(torch.FloatTensor)
ego_network_nodes = ego_network.nodes()

print(adj)
print(ego_network_nodes)
print( torch.zeros(1, 5 , 1, dtype=torch.float) )