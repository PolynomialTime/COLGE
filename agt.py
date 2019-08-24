import numpy as np
import random
import time
import os
import logging
import models
from utils.config import load_model_config

import torch.nn.functional as F
import torch


# Set up logger
logging.basicConfig(
    format='%(asctime)s:%(levelname)s:%(message)s',
    level=logging.INFO
)

"""
Contains the definition of the agent that will run in an
environment.
"""


class Agent:

    def __init__(self, id, preference, model, lr, bs):
        # agent identification
        self.id = id
        self.preference = preference

        # parameters for reinforcement learning
        self.gamma = 0.99
        self.epsilon_=1
        self.epsilon_min=0.02
        self.discount_factor =0.999990
        
        # experience replay buffer
        self.memory = []

        # set embdeding model
        self.model_name = model
        self.minibatch_length = bs
        if self.model_name == 'S2V_QN_1':
            args_init = load_model_config()[self.model_name]
            self.model = models.S2V_QN_1(**args_init)

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # configurations for S2V
        self.embed_dim = 64
        self.T = 5
        self.t = 1

    
    def reset(self, observation):
        # shrink experience dataset
        if (len(self.memory) != 0) and (len(self.memory) % 300000 == 0):
            self.memory =random.sample(self.memory,120000)
        # learning iterations
        self.iter=1

        # initialize info
        self.last_observation = observation
        self.last_action = 0
        self.last_reward = -0.01

    
    def act(self, observation):
        """Choose an action (node) from 2-hop neighbours 
        """
        # get action set from observation

        # epsilon greedy
        adj_matrix = observation[0]
        nodes_order = observation[1]
        # exploration and exploitation
        if self.epsilon_ > np.random.rand():
            index = np.random.choice( np.where( adj_matrix.numpy()[0,:,0] == 0)[0] )
            return nodes_order[index]
        else:
            info = observation[2]
            q_a = self.model(info, adj_matrix)
            q_a=q_a.detach().numpy()
            index = np.where((q_a[0, :, 0] == np.max(q_a[0, :, 0][adj_matrix.numpy()[0,:,0] == 0])))[0][0]
            return nodes_order[index]

    def reward(self, observation, action, reward):
        """Calculate loss and perform SGD using experience replay
        """
        if len(self.memory) > self.minibatch_length:
            # random sample
            (last_observation_list, action_list, reward_list, observation_list) = self.get_sample()
            # TODO: generate target tensor
            # TODO: calculate loss tensor
            # TODO: perform SGD
            

    
    """
    Experience format: (observation, action, reward, new observation)

    """

    def get_sample(self):
        """Randomly sample a batch of experiences from experience dataset
        """
        # draw a minibatch of experiences
        minibatch = random.sample(self.memory, self.minibatch_length - 1)
        minibatch.append(self.memory[-1])
        # extract information from sampled minibatch
        last_observation_list = []
        action_list = []
        reward_list = []
        observation_list = []
        last_observation_list.append(minibatch[0][0])
        action_list.append(minibatch[0][1])
        reward_list.append(minibatch[0][2])
        observation_list.append(minibatch[0][3])
        
        for last_observation_, action_, reward_, observation_ in minibatch[-self.minibatch_length + 1:]:
            last_observation_list.append(last_observation_)
            action_list.append(action_)
            reward_list.append(reward_)
            observation_list.append(observation_)

        return (last_observation_list, action_list, reward_list, observation_list)

    
    def remember(self, last_observation, last_action, last_reward, observation):
        """Store an experience to the dataset
        """
        self.memory.append((last_observation, last_action, last_reward, observation))

    
    def save_model(self):
        """Save the training model
        """
        # get current working directory
        cwd = os.getcwd()
        torch.save(self.model.state_dict(), cwd + '/' + str(self.id) + '-model.pt') 