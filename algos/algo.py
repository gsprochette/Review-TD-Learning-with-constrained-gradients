import numpy as np

"""Defines the class `Algo` : all algorithm learning at each step
should inherit from `Algo`.
"""


class Algo:
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


    def policy(self, state):
        """Decides what action to take at state `state`.
        To be defined in class instances.
        """
        pass


    def update_parameters(self, state, new_state, reward):
        """Updates the parameters according to the last step.
        To be defined in class instances.
        """
        pass


class TD0(Algo):
    """Temporal Difference algorithm TD0"""

    def __init__(self, env, mu0, epsilon, policy):
        super(TD0, self).__init__(env, mu0)
        self.epsilon = epsilon
        self.policy_fun = policy

        self.value_function = np.zeros(self.env.nstate)
        self.lr = None  # learning rate

    def policy(self, state):
        kwargs = {
            "env": self.env, "state": state,
            "V": self.value_function, "epsilon": self.epsilon
            }
        return self.policy_fun(**kwargs)

    def update_parameters(self, state, new_state, reward):
        d_tn = reward + self.value_function(new_state) \
            - self.value_function(state)

        self.value_function[state] += self.lr * d_tn
