
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
import torch.nn.functional as F
import policy


class AbstractAlgo:
    """Defines the methods common to all training algorithm
    defined in this directory.
    """

    def __init__(self, env, model, target=None, mu0=None,
                 constraint=False, residual=False):
        super(AbstractAlgo, self).__init__()

        self.env = env  # environment
        self.model = model  # model
        self.mu0 = mu0  # initial distribution

        if target is None:  # value
            self.target = None
        elif target == 'best':
            self.target = policy.best
        elif target == 'softmax':
            self.target = policy.softmax
        else:
            raise ValueError('Unknown target: "{}"'.format(target))

        self.constr = constraint  # constrains if True
        self.residual = residual
        self.lr_fun = None

        # initialize training informations
        self.nepisode = 0
        self.rewards = []
        self.optimizer = optim.SGD(self.model.parameters(), lr=1.)
        self.scheduler = None

        self.name = "Abstract algorithm"
        self.line_style = '-' if self.constr else '--'
        self.color = 'k'


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

    def plot_kwargs(self):
        kwargs = dict(
            label=self.name, ls=self.line_style, c=self.color
        )
        return kwargs


class TD0(AbstractAlgo):
    """Temporal Differences TD0 algorithm"""
    def __init__(self, env, model, pol, target=None, mu0=None,
                 constraint=False, residual=False,
                 lr_fun=None):
        super(TD0, self).__init__(
            env, model, target=None, mu0=mu0,
            constraint=constraint, residual=residual
            )
        self.name = ('R' if self.residual else '') + "TD0" \
            + (' -- Constrained' if self.constr else '')
        self.color = 'tab:purple' if self.residual else 'tab:blue'

        self.pol = pol  # policy
        self.lr_fun = self.init_lr_fun(lr_fun, None)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lr_fun)

    def policy(self):
        action_idx = self.pol(None)
        return action_idx

    def set_gradient(self, state, new_state, reward):
        v_curr = self.model(state)
        v_next = self.model(new_state)
        if not self.residual:
            v_next.detach_()  # ignore gradient of bootstrap
        td = v_curr - reward - self.env.gamma * v_next

        err = td ** 2
        err.backward()


class QLearning(AbstractAlgo):
    """Q-Learning algorithm"""

    def __init__(self, env, model, pol, target=None, mu0=None,
                 constraint=False, residual=False,
                 lr_fun=None):
        super(QLearning, self).__init__(
            env, model, target=target, mu0=mu0,
            constraint=constraint, residual=residual
            )
        self.name = ('R' if self.residual else '') + "Q-Learning" \
            + (' -- Constrained' if self.constr else '')
        self.color = 'tab:red' if self.residual else 'tab:orange'

        self.pol = pol
        self.lr_fun = self.init_lr_fun(lr_fun, None)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lr_fun)

    def policy(self):
        state = self.env.state
        if hasattr(state, '__iter__'):  # list or tuple
            state = Variable(torch.Tensor(state))
        else:
            state = Variable(torch.Tensor([state]))
        state = state.unsqueeze(0).unsqueeze(1)
        qval = self.model(state)
        qval = qval.squeeze(1).squeeze(0)
        qval = qval.data.numpy()
        action_idx = self.pol(qval)

        return action_idx

    def set_gradient(self, state, action_idx, new_state, reward):
        print(new_state)
        q_next = self.model(new_state)
        q_best_a = self.target(q_next)
        if not self.residual:
            q_best_a.detach_()  # ignore gradient of bootstrap
        q_curr = self.model(state)  # Q(s_t, a)
        td = q_curr - reward - self.env.gamma * q_best_a

        err = td ** 2
        err.backward()


class DeepQLearning(AbstractAlgo):
    """Q-Learning algorithm"""

    def __init__(self, env, model, pol, target=None, mu0=None,
                 constraint=False, residual=False,
                 lr_fun=None):
        super(DeepQLearning, self).__init__(
            env, model, target=target, mu0=mu0,
            constraint=constraint, residual=residual
            )
        self.name = ('R' if self.residual else '') \
            + "Deep Q-Learning" \
            + (' -- Constrained' if self.constr else '')
        self.color = 'tab:brown' if self.residual else 'tab:green'

        self.pol = pol
        self.lr_fun = self.init_lr_fun(lr_fun, None)
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=self.lr_fun)

    def policy(self):
        state = self.env.state
        if hasattr(state, '__iter__'):  # list or tuple
            state = Variable(torch.Tensor(state))
        else:
            state = Variable(torch.Tensor([state]))
        state = state.unsqueeze(0).unsqueeze(1)
        qval = self.model(state)
        qval = qval.squeeze(1).squeeze(0)
        qval = qval.data.numpy()
        action_idx = self.pol(qval)

        return action_idx

    def set_gradient(self, state, new_state, reward):
        q_next = self.model(new_state)
        q_best_a = self.target(q_next)
        if self.residual:
            q_best_a.detach_()  # ignore gradient of bootstrap
        q_curr = self.model(state)  # Q(s_t, a)
        td = q_curr - reward - self.env.gamma * q_best_a

        loss = self.huber_loss(td)
        loss.backward()

    @staticmethod
    def huber_loss(delta):
        delta_abs = torch.abs(delta)
        switch = (delta_abs[0] < 1).data[0]

        loss = delta_abs ** 2 / 2 if switch else delta_abs - 0.5
        return loss
