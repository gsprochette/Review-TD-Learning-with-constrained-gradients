import numpy as np
from abc import ABC


class Model(ABC):
    ''' A model specifies the way the state value function or action value
    function is approximated. At least v and grad_v or q and grad_q must be
    defined.'''
    def v(self, state):
        pass

    def q(self, state, action):
        pass

    def grad_v(self, state):
        pass

    def grad_q(self, state, action):
        pass


class LinearBaird(Model):
    def __init__(self):
        M = np.zeros((6, 7))
        # Each line i : the parameters associated to state i
        M[:, 0] = 1
        for i in range(5):
            M[i, i + 1] = 2
        M[5, 0] = 2
        M[5, 6] = 1
        self.M = M
        self.n_params = 7

    def v(self, state, theta):
        return self.M[state] @ theta

    def all_v(self, theta):
        return self.M @ theta

    def grad_v(self, state):
        return self.M[state]
