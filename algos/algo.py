
"""Defines the class `Algo` : all algorithm learning at each step
should inherit from `Algo`.
"""
from abc import ABC, abstractmethod


class Algo(ABC):
    """Defines the methods common to all training algorithm
    defined in this directory.
    """

    def __init__(self, env, mu0=None):
        super(Algo, self).__init__()

        self.env = env  # environment
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
