
"""Defines the class `Algo` : all algorithm learning at each step
should inherit from `Algo`.
"""
import numpy as np
import numpy.random as npr
import numpy.linalg as LA
from abc import ABC, abstractmethod
from envs.env import Baird

class Algo(ABC):
    """Defines the methods common to all training algorithm
    defined in this directory.
    """

    def __init__(self, env, mod, mu0=None):
        super(Algo, self).__init__()

        self.env = env  # environment
        self.mod = mod
        self.mu0 = mu0  # initial distribution

        # initialize training informations
        self.nepisode = 0
        self.rewards = []

    def episode(self):
        """Trains on one full episode"""
        self.env.reset(self.mu0)
        reward_acc = []
        stop = False
        while not stop:
            state = self.env.state
            action = self.policy()
            reward, stop = self.env.step(action)
            self.update_parameters(state, self.env.state, reward)
            reward_acc.append(reward)
        self.nepisode += 1
        self.rewards.append(reward_acc)

    def policy(self):
        """Decides what action to take at state `state`.
        To be defined in class instances.
        """
        if isinstance(self.env, Baird):
            return 0
        else:
            raise NotImplementedError(
                'Residual gradient not implemented for' \
                + 'this environment')

    @abstractmethod
    def update_parameters(self, state, new_state, reward):
        """Updates the parameters according to the last step.
        To be defined in class instances.
        """
        pass


class TD0(Algo):

    def __init__(self, env, mod, mu0=None, epsilon=None):
        super(TD0, self).__init__(env, mod, mu0)
        self.epsilon = epsilon

    def update_parameters(self, s, new_s, r, theta, alpha):
        ''' theta (arr): set of parameters to estimate the value function
            alpha (double): learning rate.
        '''
        estimate = r + self.env.gamma * self.mod.v(new_s, theta)
        delta = self.mod.v(s, theta) - estimate
        theta -= 2 * alpha * delta * self.mod.grad_v(s)
        return theta


class QLearning(Algo):
    def __init__(self, env, mu0=None):
        super().__init(env, mu0)


class ResidualGradient(Algo):
    def __init__(self, env, mod, mu0=None, phi=0.4):
        super().__init__(env, mod, mu0)
        self.phi = phi

    def update_parameters(self, s, new_s, r, theta, alpha):
        ''' theta (arr): set of parameters to estimate the value function
            alpha (double): learning rate.
        '''
        model = self.mod
        g = self.env.gamma
        delta = r + g * model.v(new_s, theta) - model.v(s, theta)
        grad = self.phi * g * model.grad_v(new_s) - model.grad_v(s)
        theta -= 2 * alpha * delta * grad
        return theta


class ConstrainedGradient(Algo):
    def update_parameters(self, s, new_s, r, theta, alpha):
        ''' theta (arr): set of parameters to estimate the value function
            alpha (double): learning rate.
        '''
        estimate = r + self.env.gamma * self.mod.v(new_s, theta)
        delta = self.mod.v(s, theta) - estimate
        grad = self.mod.grad_v(s)
        bootstrap_grad = self.mod.grad_v(new_s)
        boot_normalized = bootstrap_grad / LA.norm(bootstrap_grad)
        proj = np.dot(grad, boot_normalized) * boot_normalized
        constrained_grad = grad - proj
        theta -= 2 * alpha * delta * constrained_grad
        return theta


class ConstrainedResidualGradient(Algo):
    def update_parameters(self, s, new_s, r, theta, alpha):
        ''' theta (arr): set of parameters to estimate the value function
            alpha (double): learning rate.
        '''
        estimate = r + self.env.gamma * self.mod.v(new_s, theta)
        delta = self.mod.v(s, theta) - estimate
        grad = self.mod.grad_v(s) - self.env.gamma * self.mod.grad_v(new_s)
        bootstrap_grad = self.mod.grad_v(new_s)
        boot_normalized = bootstrap_grad / LA.norm(bootstrap_grad)
        proj = np.dot(grad, boot_normalized) * boot_normalized
        constrained_grad = grad - proj
        theta -= 2 * alpha * delta * constrained_grad
        return theta


class GTD2(Algo):
    def __init__(self, env, mod, mu0=None):
        ''' Warning : this model is only valid for a linear model.
        beta (double): learning rate for w.'''
        super().__init__(env, mod, mu0)
        self.w = np.zeros(mod.n_params)

    def update_parameters(self, s, new_s, r, theta, alpha, beta=None):
        ''' theta (arr): set of parameters to estimate the value function
            alpha (double): learning rate.
        '''
        if beta is None:
            beta = alpha
        g = self.env.gamma
        model = self.mod
        delta = r + g * model.v(new_s, theta) - model.v(s, theta)

        phi_s, phi_snew = model.grad_v(s), model.grad_v(new_s)
        grad = phi_s - self.env.gamma * phi_snew

        theta += alpha * (phi_s.T @ self.w) * grad
        self.w += beta * (delta - phi_s.T @ self.w) * phi_s
        return theta
