import numpy as np
from collections import namedtuple
import numbers
import gridrender as gui
from tkinter import Tk
import tkinter.font as tkfont
import copy


class Env:
    """Abstract environment class. All discrete environment should inherit from
    this class and define `nstate`, `naction`, `reward`, `transition` and 
    the method `is_terminal`.
    """

    def __init__(self):
        self.nstate = None  # n
        self.naction = None  # a

        self.reward = None  # n*a
        self.transition = None  # n*a*n

    def reset(self, mu0):
        assert len(mu0) == self.nstate
        state = np.random.choice(self.nstate, p=mu0)
        return state

    def available_actions(self, state):
        all_actions = self.transition[state, :, :]
        is_available = np.sum(all_actions, 1)
        return is_available > 0

    def step(self, state, action):
        trans_proba = self.transition[state, action, :]
        assert np.sum(trans_proba) > 0
        next_state = np.random.choice(self.nstate, p=trans_proba)
        reward = self.reward[state, action]
        stop = self.is_terminal(state, action)
        
        return next_state, reward, stop

    def is_terminal(self, state, action):
        pass


class GridWorld(Env):

    def __init__(self, gridh, gridl, terminal_state):
        super(GridWorld, self).__init__()
        self.height = gridh
        self.length = gridl
        self.nstate = gridh * gridl

        self.actions = ['up', 'left', 'down', 'right']
        self.naction = 4

        self.reward = np.zeros(self.nstate)
        self.reward[*self.lin2matrix(terminal_state)] = 1

        self.init_transition()
        self.terminal_state = self.matrix2lin(*terminal_state)

    def matrix2lin(self, coord1, coord2):
        return coord1 * self.length + coord2

    def lin2matrix(self, coord):
        coord1 = int(coord / self.length)
        coord2 = coord % self.length
        return coord1, coord2

    def init_transition(self)
        self.transition = np.zeros(self.nstate, self.naction, self.nstate)
        deltax = [-1, 0, 1, 0]
        deltay = [0, -1, 0, 1]

        for i in range(self.nstate):
            x0, y0 = self.lin2matrix(i)
            for a in range(self.naction):
                x1, y1 = x0 + deltax[a], y0 + deltay[a]
                if x1 >= 0 and x1 < self.height \
                    and y1 >= 0 and y1 < self.length:
                    j = self.matrix2lin(x1, y1)
                    self.transition[i, a, j] = 1

    def is_terminal(state, _):
        return state == self.terminal_state
