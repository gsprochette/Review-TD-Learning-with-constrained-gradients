
"""Defines the class `Algo` : all algorithm learning at each step
should inherit from `Algo`.
"""
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


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

    def episode(self):
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
    def policy(self, state):
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

    @abstractmethod
    def constraint_dir(self, state, new_state, reward):
        """Computes the projection vector $\hat g_v(s_{t+1})$ as
        defined in https://openreview.net/pdf?id=Bk-ofQZRb
        """
        pass

    def update_parameters(self, state, new_state, reward):
        """Computes gradient step and projects it if constrained"""

        g_td = None
        if self.constr:
            g_v = self.constraint_dir(state, new_state, reward)
            pg_td = torch.dot(g_td, g_v) * g_v
            g_td = g_td - pg_td

        # update parameters
        return


class TD0(Algo):

    def __init__(self, env, mu0, epsilon):
        super(TD0, self).__init__(env, mu0)
        self.epsilon = epsilon
        # self.statevalue =

    def policy(self, state):
        pass


class QLearning(Algo):
    def __init__(self):
        pass

    def policy(self, state):
        pass


class ResidualGradient(Algo):
    def __init__(self):
        pass

    def policy(self, state):
        pass
