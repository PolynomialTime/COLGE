import torch
import numpy as np
import networkx as nx
import graph
import random


g = graph.Graph('cycle_graph', 100, 1)
ego_network = g.two_level_ego_network(1)
adj = nx.adjacency_matrix(ego_network)
adj = adj.todense() # store in matrix
adj = torch.from_numpy(np.expand_dims(adj.astype(int), axis=0))
adj = adj.type(torch.FloatTensor)
ego_network_nodes = ego_network.nodes()

#print(adj.numpy()[0,:,0])
#x = np.where(adj.numpy()[0,:,0] == 0)
#print(x)
#print(adj)
#print(ego_network_nodes)
#print( torch.zeros(1, 5 , 1, dtype=torch.float) )


info = torch.zeros(1, 5, 2, dtype=torch.float)
info[0,0,0] = 1
print(info)