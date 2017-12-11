
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

    @abstractmethod
    def policy(self):  # Could be outside the class
        """Decides what action to take at state `state`.
        To be defined in class instances.
        """
        pass

    @abstractmethod
    def loss(self, state, new_state, reward):
        """Computes the loss corresponding to the algorithm
        implemented, e.g. for epsilon-step Q-learning:
        $$
            L = \| q(s_t, a_t|\theta) - r_t - \gamma \max_a q(s_{t+1}, a|\theta) \|^2
        $$
        """
        pass

    def update(self, state, new_state, reward, optimizer):
        """Computes gradient step and projects it if constrained. Returns current loss and parameter update"""

        self.model.zero_grad()
        err = self.loss(state, new_state, reward)
        err.backward()

        if self.constr:
            g_vs = self.model.g_v(new_state)
            for param, g_v in zip(self.model.parameters(), g_vs):
                param.grad -= torch.dot(param.grad, g_v) * g_v
        optimizer.step()


class TD0(Algo):
    """Temporal Differences TD0 algorithm"""
    def __init__(self, env, model, mu0=None, constraint=False):
        super(TD0, self).__init__(
            env, model, mu0=None, constraint=constraint
            )

    def policy(self):
        pass

    def loss(self, state, new_state, reward):
        pass


class QLearning(Algo):
    """Q-Learning algorithm"""

    def policy(self):
        pass

    def loss(self, state, new_state, reward):
        pass


class ResidualGradient(Algo):
    """Residual Gradient algorithm"""

    def __init__(self, env, model, policy,
                 mu0=None, constraint=False):
        super(ResidualGradient, self).__init__(
            env, model, mu0=None, constraint=constraint
            )
        self.pol = policy

    def policy(self):
        return self.pol(self.env.state)

    def loss(self, state, new_state, reward):
        q_next = self.model.forward(new_state)
        q_best_a = self.max_value(q_next)
        expected = self.env.gamma * q_best_a + reward
        q_curr = self.model.forward(state)
        td = q_curr - expected

        return td ** 2

    @staticmethod
    def max_value(qval):
        """V approximation through Q : $V(s) = \max_a Q(s, a)$"""

        if qval.numel() == 1:  # value function
            return qval
        return torch.max(qval)
