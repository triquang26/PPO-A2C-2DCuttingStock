import gym_cutting_stock
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
# from policy import GreedyPolicy, RandomPolicy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from policy import GreedyPolicy, RandomPolicy
from .ProximalPolicyOptimization import ProximalPolicyOptimization
from .A2C import ActorCriticPolicy2
from .EpisodeEvaluator import EpisodeEvaluator
import signal
import sys

class Policy2353335:
    def __init__(self, policy_id):
        self.policy_id = policy_id
        self.training = False
        self.policy = None
        if (policy_id == 1):
            self.policy = ActorCriticPolicy2()
            self.policy.load_model('saved_models/model_a2c_best.pt')
        if policy_id == 2:
            self.policy = ProximalPolicyOptimization()
            self.policy.load_model('saved_models/model_a2c_best.pt')
        self.policy.training = False
            

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)