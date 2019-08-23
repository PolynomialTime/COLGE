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
        # learning iterations
        self.iter=1

        # initial observation
        self.observation = observation

    
    def act(self, observation):
        """Choose an action (node) from 2-hop neighbours 
        """
        # get action set from observation

        # epsilon greedy
        if self.epsilon_ > np.random.rand():
            