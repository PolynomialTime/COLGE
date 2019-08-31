import argparse
import agent
import environment
import runner
import graph
import logging
import numpy as np
import networkx as nx
import sys

# # 2to3 compatibility
# try:
#     input = raw_input
# except NameError:
#     pass

# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

parser = argparse.ArgumentParser(description='RL running machine')
parser.add_argument('--termination_step', type=int, metavar='n', default=5, help='termination step')
parser.add_argument('--environment_name', metavar='ENV_CLASS', type=str, default='community', help='Class to use for the environment. Must be in the \'environment\' module')
parser.add_argument('--graph_type',metavar='GRAPH', default='cycle_graph',help ='Type of the initial graph')
parser.add_argument('--model', type=str, default='S2V_QN_1', help='model name')
parser.add_argument('--episodes', type=int, metavar='nepoch',default=500000, help="number of episdoes")
parser.add_argument('--lr',type=float, default=1e-4,help="learning rate")
parser.add_argument('--bs',type=int,default=16,help="minibatch size for training")
parser.add_argument('--node_nbr', type=int, metavar='nnode',default=100, help="number of node in generated graphs")


def main():
    args = parser.parse_args()
    logging.info('Loading graph %s ...' % args.graph_type)
    g = graph.Graph(args.graph_type, args.node_nbr)

    logging.info('Loading environment %s ...' % args.environment_name)
    env = environment.Environment(g, args.environment_name)
    
    logging.info('Loading agents...')
    agents = []
    for i in range(args.node_nbr):
        agents.append(agent.Agent(i, env.preferences[i], args.model, args.lr, args.bs, args.termination_step))


    print("Running a simulation...")
    my_runner = runner.Runner(env, agents, args.termination_step, args.environment_name)
    my_runner.loop(args.episodes, args.termination_step)

if __name__ == "__main__":
    main()
