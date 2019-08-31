import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import agent
import environment
import copy
import os

multiprocessing.set_start_method('spawn', True)
"""
This is the machinnery that runs agents in an environment.

"""

out_dir = '/content/drive/My Drive/'

class Runner:
    def __init__(self, env, agents, termination_step, environment_name):
        self.env = env
        self.agents = agents
        self.termination_step = termination_step
        self.environment_name = environment_name
        if self.environment_name == 'community':
            if os.path.exists(str(self.termination_step) + "_community_info_.txt"):
                os.remove(str(self.termination_step) + "_community_info_.txt")
            self.info_file = open(str(self.termination_step) + "_community_info_.txt", 'a')
        
        if self.environment_name == 'small-world':
            if os.path.exists(str(self.termination_step) + "_sw_info_.txt"):
                os.remove(str(self.termination_step) + "_sw_info_.txt")
            self.info_file = open(str(self.termination_step) + "_sw_info_.txt", 'a')


    def step(self):
        observations = copy.deepcopy(self.env.observe())
        # parallely compute actions
        actions = []
        for i in range(len(self.agents)):
            actions.append(self.agents[i].act(observations[i]))
        #pool = multiprocessing.Pool(processes=6)
        #results = []
        #for i in range(0, len(self.agents)):
        #    results.append(pool.apply_async(self.agents[i].act , (observations[i],)))
        #pool.close() # close pool
        #pool.join() # wait for all jobs in the pool finished
        #for res in results:
        #    actions.append(res.get())
        # update graph
        self.env.update(actions)
        # compute rewards
        rewards = self.env.act(actions) 
        # parallelly perform SGD
        #pool = multiprocessing.Pool(processes=6)
        #for i in range(0, len(self.agents)):
        #    pool.apply_async(self.agents[i].reward, ((observations[i], actions[i], rewards[i]),))
        #pool.close()
        #pool.join()
        
        for i in range(len(self.agents)):
            self.agents[i].reward(observations[i], actions[i], rewards[i])
        
        #return (observations, actions, rewards)
        
    def loop(self, nbr_episode, termination_step):
        for episode in range(nbr_episode):
            print(" -> episode : " + str(episode))
            # initialize environment
            self.env.reset()
            # initialize agents
            for a in self.agents:
                a.reset()
            for t in range(termination_step):
                #print("     -> step : " + str(t))
                # learning
                #(observations, actions, rewards) = self.step()
                #print("        -> actions : ", actions)
                self.step()

            if episode % 5 == 0:
                if self.environment_name == 'community':
                    self.info_file.writelines(str(self.env.community_info()))
                    self.info_file.write('\n')
                if self.environment_name == 'small_world':
                    self.info_file.writelines(str(self.env.small_world_info()))
                    self.info_file.write('\n')


        
        

        