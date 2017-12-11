
"""Defines the class `Algo` : all algorithm learning at each step
should inherit from `Algo`.
"""
import numpy as np
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
    def update_parameters(self, state, new_state, reward):
        """Updates the parameters according to the last step.
        To be defined in class instances.
        """
        pass


class TD0(Algo):

    def __init__(self, env, mod, mu0=None, epsilon=None):
        super(TD0, self).__init__(env, mod, mu0)
        self.epsilon = epsilon

    def policy(self, state):
        if isinstance(self.env, Baird):
            return 0
        pass

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

    def policy(self, state):
        if isinstance(self.env, Baird):
            return 0
        pass


class ResidualGradient(Algo):
    def __init__(self, env, mod, mu0=None, epsilon=None):
        super().__init__(env, mod, mu0)

    def policy(self, state):
        if isinstance(self.env, Baird):
            return 0
        else:
            raise NotImplementedError('Residual gradient not implemented',
                                      'for this environment')

    def update_parameters(self, s, new_s, r, theta, alpha):
        ''' theta (arr): set of parameters to estimate the value function
            alpha (double): learning rate.
        '''
        estimate = r + self.env.gamma * self.mod.v(new_s, theta)
        delta = self.mod.v(s, theta) - estimate
        grad = self.mod.grad_v(s) - self.env.gamma * self.mod.grad_v(new_s)
        theta -= 2 * alpha * delta * grad
        return theta


class ConstrainedGradient(Algo):
    def __init__(self, env, mod, mu0=None, epsilon=None):
        super().__init__(env, mod, mu0)

    def policy(self, state):
        if isinstance(self.env, Baird):
            return 0
        else:
            raise NotImplementedError('Residual gradient not implemented',
                                      'for this environment')

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
