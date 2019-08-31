import numpy as np
import random
import time
import os
import logging
import copy
import model as M

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

    def __init__(self, id, preference, model_name, lr, bs, termination_step):
        #self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        # agent identification
        self.id = id
        self.preference = preference
        self.termination_step = termination_step

        # parameters for reinforcement learning
        self.gamma = 0.99
        self.epsilon_ = 1
        self.epsilon_min = 0.02
        self.discount_factor = 0.999990

        # experience replay buffer
        self.memory = []

        # set embdeding model
        self.minibatch_length = bs
        if model_name == 'S2V_QN_1':
            #args_init = load_model_config()[self.model_name]
            self.model = M.S2V_QN_1(32, 32, 0, 0).to(self.device)

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # configurations for S2V
        self.embed_dim = 32
        self.T = 4
        

    def reset(self):
        # shrink experience dataset
        if (len(self.memory) != 0) and (len(self.memory) % 300000 == 0):
            self.memory = random.sample(self.memory, 120000)
        # learning iterations
        self.t = 0

        # initialize internal info
        self.last_observation = 0
        self.last_action = 0
        self.last_reward = 0

    def act(self, observation):
        """Choose an action (node) from 2-hop neighbours 
        """
        # epsilon greedy
        adj_matrix = observation[0]
        nodes_order = observation[1]
        # exploration and exploitation
        if self.epsilon_ > np.random.rand():
            try:
                index = np.random.choice(np.where(adj_matrix.numpy()[0, :, 0] == 0)[0])
            except:
                index = 0
            return nodes_order[index]
        else:
            info = observation[2]
            q_a = self.model(info, adj_matrix)
            q_a = q_a.detach().numpy()
            index = np.where((q_a[0, :, 0] == np.max(
                q_a[0, :, 0][adj_matrix.numpy()[0, :, 0] == 0])))[0][0]
            return nodes_order[index]

    def reward(self, observation, action, reward):
        """Calculate loss and perform SGD using experience replay
        """
        if len(self.memory) > self.minibatch_length:
            # random sample
            (last_observation_list, action_list, reward_list,
             observation_list) = self.get_sample()

            for i in range(self.minibatch_length):
             # generate target tensor
                adj = observation_list[i][0]
                info = observation_list[i][2]
                step = observation_list[i][3]

                reward_ = torch.Tensor([[reward_list[i]]]).to(self.device)
                #action_ = torch.Tensor([action_list[i]]).type(torch.LongTensor)
                action_ = action_list[i]

                if step == self.termination_step - 1:
                    max_q = torch.zeros(1, 1, 1, dtype=torch.float).to(self.device)
                else:
                    max_q = torch.max(self.model(info, adj), dim=1)[0].to(self.device)

                target = reward_ + self.gamma * max_q

                # genrate target_f
                adj = last_observation_list[i][0]
                nodes_order = last_observation_list[i][1]
                info = last_observation_list[i][2]
                target_f = self.model(info, adj).to(self.device)
                #print(target_f)

                # generate target_p
                target_p = target_f.clone().to(self.device)

                # calculate loss tensor
                index = nodes_order.index(action_)
                target_f[0, index, :] = target
                loss = self.criterion(target_p, target_f)

                # perform SGD
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # adjust epsilon greedy
            if self.epsilon_ > self.epsilon_min:
                self.epsilon_ *= self.discount_factor

            # remember experience
        if self.t > 0:
            self.remember(self.last_observation, self.last_action, self.last_reward, copy.deepcopy(observation))

        # update internal information
        self.t += 1
        self.last_action = action
        self.last_observation = copy.deepcopy(observation)
        self.last_reward = reward

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

        for (last_observation_, action_, reward_, observation_) in minibatch:
            last_observation_list.append(last_observation_)
            action_list.append(action_)
            reward_list.append(reward_)
            observation_list.append(observation_)

        return (last_observation_list, action_list, reward_list, observation_list)

    def remember(self, last_observation, last_action, last_reward, observation):
        """Store an experience to the dataset
        """
        self.memory.append(
            (last_observation, last_action, last_reward, observation))

    def save_model(self):
        """Save the training model
        """
        # get current working directory
        cwd = os.getcwd()
        torch.save(self.model.state_dict(), cwd + '/' + str(self.id) + '-model.pt')


"""Following functions are for parallel computing use
"""


"""
def reward(agent:Agent, observation, action, reward):
    # Calculate loss and perform SGD using experience replay
    

    if len(agent.memory) > agent.minibatch_length:
        # random sample
        (last_observation_list, action_tens, reward_tens, observation_list) = agent.get_sample()

        # generate target tensor
        adj = observation_list[0][0]
        info = observation_list[0][2]
        step = observation_list[0][3]

        if step == agent.termination_step - 1:
            max_q_tens = torch.zeros(1, 1, 1, dtype=torch.float)
        else:
            max_q_tens = torch.max(agent.model(info, adj), dim=1)[0]

        for i in range(1, agent.minibatch_length):
            adj = observation_list[i][0]
            info = observation_list[i][2]
            step = observation_list[0][3]
            if step == agent.termination_step - 1:
                max_q_tens = torch.cat((max_q_tens, orch.zeros(1, 1, 1, dtype=torch.float)))
            else:
                max_q_tens = torch.cat((max_q_tens, torch.max(agent.model(info, adj), dim=1)[0]))
        print('zzzzz')
        target = reward_tens + agent.gamma * max_q_tens
        #if agent.id == 0:
        #    print(target)

        # genrate target_f
        adj = last_observation_list[0][0]
        info = last_observation_list[0][2]
        target_f = agent.model(info, adj)

        for i in range(1, agent.minibatch_length):
            adj = last_observation_list[i][0]
            info = last_observation_list[i][2]
            target_f = torch.cat((target_f, agent.model(info, adj)))

        # generate target_p
        target_p = target_f.clone()

        # calculate loss tensor
        target_f[range(agent.minibatch_length), action_tens, :] = target
        loss = agent.criterion(target_p, target_f)

        # perform SGD
        agent.optimizer.zero_grad()
        loss.backward()
        agent.optimizer.step()

        # adjust epsilon greedy
        if agent.epsilon_ > agent.epsilon_min:
                agent.epsilon_ *= agent.discount_factor

    # remember experience
    if agent.t > 0:
        agent.remember(agent.last_observation, agent.last_action, agent.last_reward, copy.deepcopy(observation))

    # update internal information
    agent.t = agent.t + 1
    agent.last_action = action
    agent.last_observation = copy.deepcopy(observation)
    agent.last_reward = reward
"""

