import matplotlib.pyplot as plt
import numpy as np
import agt
import env

"""
This is the machinnery that runs agents in an environment.

"""

class Runner:
    def __init__(self, environment:env.Environment, agents, verbose=False):
        self.environment = environment
        self.agents = agents
        self.verbose = verbose

    def step(self):
        observations = self.environment.observe()
        