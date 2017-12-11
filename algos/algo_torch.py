
"""Defines the class `Algo` : all algorithm learning at each step
should inherit from `Algo`.
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter


class Algo(ABC):
    """Defines the methods common to all training algorithm
    defined in this directory.
    """

    def __init__(self, env, model, mu0=None, constraint=False):
        super(Algo, self).__init__()

        self.env = env  # environment
        self.model = model  # model
        self.mu0 = mu0  # initial distribution
        self.constr = constraint  # constrains if True

        # initialize training informations
        self.nepisode = 0
        self.rewards = []

    def episode(self):  # Should be in main.py
        """Trains on one full episode"""
        state = self.env.reset(self.mu0)
        reward_acc = []
        stop = False
        while not stop:
            action = self.policy(state)
            new_state, reward, stop = self.env.step(state, action)
            self.update_parameters(state, new_state, reward)
            reward_acc.append(reward)
            state = new_state
        self.nepisode += 1
        self.rewards.append(reward_acc)

    @abstractmethod
    def policy(self, state):  # Could be outside the class
        """Decides what action to take at state `state`.
        To be defined in class instances.
        """
        pass

    @abstractmethod
    def loss(self, state, new_state, reward):
        """Computes the loss corresponding to the algorithm
        implemented, e.g. for Q-learning :
        $$
            L = \| q(s_t, a_t|\theta) - r_t - \gamma \max_a q(s_{t+1}, a|\theta) \|^2
        $$
        """
        pass

    def update(self, state, new_state, reward):
        """Computes gradient step and projects it if constrained. Returns current loss and parameter update"""

        self.model.zero_grad()
        err = self.loss(state, new_state, reward)
        err.backward()
        g_tds = [param.grad for param in self.model.parameters()]

        if not self.constr:
            return g_tds

        g_vs = self.model.g_v(new_state)
        g_updates = [
            g_td - torch.dot(g_td, g_v) * g_v
            for g_td, g_v in zip(g_tds, g_vs)
        ]
        return g_updates


class TD0(Algo):
    """Temporal Differences TD0 algorithm"""
    def __init__(self, env, mu0, epsilon):
        super(TD0, self).__init__()

    def policy(self, state):
        pass

    def loss(self, state, new_state, reward):
        pass


class QLearning(Algo):
    """Q-Learning algorithm"""
    def __init__(self):
        super(QLearning, self).__init__()

    def policy(self, state):
        pass

    def loss(self, state, new_state, reward):
        pass


class ResidualGradient(Algo):
    """Residual Gradient algorithm"""
    def __init__(self):
        super(ResidualGradient, self).__init__()

    def policy(self, state):
        pass

    def loss(self, state, new_state, reward):
        pass
