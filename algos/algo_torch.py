
"""Defines the class `Algo` : all algorithm learning at each step
should inherit from `Algo`.
"""
from abc import ABC, abstractmethod
from warnings import warn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter


class AbstractAlgo:
    """Defines the methods common to all training algorithm
    defined in this directory.
    """

    def __init__(self, env, model, mu0=None, constraint=False):
        super(AbstractAlgo, self).__init__()

        self.name = "Abstract algorithm"
        self.env = env  # environment
        self.model = model  # model
        self.mu0 = mu0  # initial distribution

        self.constr = constraint  # constrains if True
        self.lr_fun = None

        # initialize training informations
        self.nepisode = 0
        self.rewards = []
        self.optimizer = optim.SGD(self.model.parameters(), lr=1.)
        self.scheduler = None

    def update(self, state, new_state, reward):
        """Computes gradient step and projects it if constrained.
        Returns current loss and parameter update.
        """

        self.model.zero_grad()
        self.set_gradient(state, new_state, reward)

        if self.constr:
            g_vs = self.model.g_v(new_state)
            for param, g_v in zip(self.model.parameters(), g_vs):
                param.grad -= torch.dot(param.grad, g_v) * g_v
        self.optimizer.step()

    def init_lr_fun(self, lr_fun, builtin_lr_fun):
        """Resolves user-input and built-in conflict"""

        if builtin_lr_fun is not None and lr_fun is not None:
            warn("{} has a built-in".format(self.name) \
            + " learning rate update, user learning rate" \
            + " is being ignored")
            return builtin_lr_fun
        elif builtin_lr_fun is not None:
            return builtin_lr_fun
        elif lr_fun is not None:
            return lr_fun
        warn("No learning rate provided for {}".format(self.name))
        return None

    def set_gradient(self, state, new_state, reward):
        """Used when the update does not derive from a Loss function.
        All parameters' gradients will be set with the value returned
        by this function.
        Output should be a list of Variables, with shapes matching
        self.model.parameters()
        """
        # for param, grad in zip(self.model.parameters(), grads):
        #     param.grad = grad
        pass

    def policy(self):
        """Decides what action to take at state `state`.
        To be defined in class instances.
        """
        raise NotImplementedError

    @staticmethod
    def max_value(qval):
        """V approximation through Q : $V(s) = \max_a Q(s, a)$"""

        if qval.numel() == 1:  # value function
            return qval
        return torch.max(qval)


class TD0(AbstractAlgo):
    """Temporal Differences TD0 algorithm"""
    def __init__(self, env, model, policy, mu0=None,
                 constraint=False, lr_fun=None):
        super(TD0, self).__init__(
            env, model, mu0=None, constraint=constraint,
            )
        self.pol = policy
        lr_fun = self.init_lr_fun(lr_fun, None)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lr_fun)

    def policy(self):
        return self.pol(self.env.state)

    def set_gradient(self, state, new_state, reward):
        v_curr = self.model(state)
        v_next = self.model(new_state)
        td = v_curr - reward - self.env.gamma * v_next

        v_curr.backward()
        for param in self.model.parameters():
            param.grad *= 2 * td


class ResidualTD0(AbstractAlgo):
    """Residual TD0 algorithm"""
    def __init__(self, env, model, policy, mu0=None,
                 constraint=False, lr_fun=None):
        super(ResidualTD0, self).__init__(
            env, model, mu0=None, constraint=constraint,
            )
        self.pol = policy
        lr_fun = self.init_lr_fun(lr_fun, None)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lr_fun)

    def policy(self):
        return self.pol(self.env.state)

    def set_gradient(self, state, new_state, reward):
        v_curr = self.model(state)
        v_next = self.model(new_state)
        td = v_curr - reward - self.env.gamma * v_next

        err = td ** 2
        err.backward()


############################################################################
######  Q-Learning Methods could have a generic Q(s, .) -> \hat Q(s)  ######
############################################################################


class QLearning(AbstractAlgo):
    """Q-Learning algorithm"""

    def __init__(self, env, model, policy, mu0=None,
                 constraint=False, lr_fun=None):
        super(QLearning, self).__init__(
            env, model, mu0=None, constraint=constraint,
            )
        self.pol = policy
        self.name = "Q-Learning" + (' -- Constrained' if self.constr else '')

        lr_fun = self.init_lr_fun(lr_fun, None)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lr_fun)

    def policy(self):
        return self.pol(self.env.state)

    def set_gradient(self, state, new_state, reward):
        q_next = self.model(new_state)
        q_best_a = self.max_value(q_next)
        q_curr = self.model(state)  # Q(s_t, a)
        td = q_curr - reward - self.env.gamma * q_best_a

        q_curr.backward()
        for param in self.model.parameters():
            param.grad *= 2 * td


class ResidualQLearning(AbstractAlgo):
    """Residual Gradient algorithm"""

    def __init__(self, env, model, policy, mu0=None,
                 constraint=False, lr_fun=None):
        super(ResidualQLearning, self).__init__(
            env, model, mu0=None, constraint=constraint,
            )
        self.pol = policy
        self.name = "Residual Q-Learning" + (' -- Constrained' if self.constr else '')

        lr_fun = self.init_lr_fun(lr_fun, None)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lr_fun)

    def policy(self):
        return self.pol(self.env.state)

    def set_gradient(self, state, new_state, reward):
        q_next = self.model(new_state)
        q_best_a = self.max_value(q_next)
        q_curr = self.model(state)
        td = q_curr - reward - self.env.gamma * q_best_a  # delta

        err = td ** 2
        err.backward()
