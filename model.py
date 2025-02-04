import torch
import torch.nn.functional as F
import random
#from gensim.models import Word2Vec
import networkx as nx
import numpy as np

class S2V_QN_1(torch.nn.Module):
    def __init__(self, reg_hidden, embed_dim, len_pre_pooling, len_post_pooling, T=4):

        super(S2V_QN_1, self).__init__()
        self.T = T
        self.embed_dim=embed_dim
        self.reg_hidden=reg_hidden
        self.len_pre_pooling = len_pre_pooling
        self.len_post_pooling = len_post_pooling
        #self.mu_1 = torch.nn.Linear(1, embed_dim)
        #torch.nn.init.normal_(self.mu_1.weight,mean=0,std=0.01)

        self.mu_1 = torch.nn.Parameter(torch.Tensor(1, embed_dim)) # theta_1
        torch.nn.init.normal_(self.mu_1, mean=0, std=0.01)
        self.mu_2 = torch.nn.Linear(embed_dim, embed_dim,True) # theta_2
        torch.nn.init.normal_(self.mu_2.weight, mean=0, std=0.01)

        self.list_pre_pooling = []
        for i in range(self.len_pre_pooling):
            pre_lin = torch.nn.Linear(embed_dim,embed_dim,bias=True)
            torch.nn.init.normal_(pre_lin.weight, mean=0, std=0.01)
            self.list_pre_pooling.append(pre_lin)

        self.list_post_pooling = []
        for i in range(self.len_post_pooling):
            post_lin =torch.nn.Linear(embed_dim,embed_dim,bias=True)
            torch.nn.init.normal_(post_lin.weight, mean=0, std=0.01)
            self.list_post_pooling.append(post_lin)

        self.q_1 = torch.nn.Linear(embed_dim, embed_dim,bias=True)
        torch.nn.init.normal_(self.q_1.weight, mean=0, std=0.01)
        self.q_2 = torch.nn.Linear(embed_dim, embed_dim,bias=True)
        torch.nn.init.normal_(self.q_2.weight, mean=0, std=0.01)

        if self.reg_hidden > 0:
            self.q_reg = torch.nn.Linear(2 * embed_dim, self.reg_hidden)
            torch.nn.init.normal_(self.q_reg.weight, mean=0, std=0.01)
            self.q = torch.nn.Linear(self.reg_hidden, 1)
        else:
            self.q = torch.nn.Linear(2 * embed_dim, 1)
        torch.nn.init.normal_(self.q.weight, mean=0, std=0.01)

    def forward(self, xv, adj): # input: preference weight vector; adjacent matrix of a 2-level ego network (observation)

        minibatch_size = xv.shape[0]
        nbr_node = xv.shape[1]
        #print(self.mu_1)

        for t in range(self.T):
            if t == 0:
                mu = torch.matmul(xv, self.mu_1).clamp(0) # nbr x dim
            else:
                #mu_1 = self.mu_1(xv).clamp(0)
                mu_1 = torch.matmul(xv, self.mu_1).clamp(0) # nbr x dim
                for i in range(self.len_pre_pooling):
                    mu = self.list_pre_pooling[i](mu).clamp(0) # nbr x dim
                mu_pool = torch.matmul(adj, mu) # nbr x dim

                # after pooling
                for i in range(self.len_post_pooling):
                    mu_pool = self.list_post_pooling[i](mu_pool).clamp(0) # dim x dim

                mu_2 = self.mu_2(mu_pool) # dim x dim
                mu = torch.add(mu_1, mu_2).clamp(0) # nbr x dim

        q_1 = self.q_1(torch.matmul(xv.transpose(1,2),mu)).expand(minibatch_size,nbr_node,self.embed_dim)
        q_2 = self.q_2(mu)
        #print(q_2)
        q_ = torch.cat((q_1, q_2), dim=-1) # concatenate by column
        if self.reg_hidden > 0:
            q_reg = self.q_reg(q_).clamp(0)
            q = self.q(q_reg)
        else:
            q_=q_.clamp(0)
            q = self.q(q_) # q values
        #print(q)
        return q
