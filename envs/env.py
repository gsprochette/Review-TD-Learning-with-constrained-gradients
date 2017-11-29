import numpy as np
from collections import namedtuple
import numbers
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

    def reset(self, mu0=None):
        assert mu0 is None or len(mu0) == self.nstate
        state = np.random.choice(self.nstate, p=mu0)
        return state

    def available_actions(self, state):
        ''' Returns the indices of all available actions from state `state` '''
        all_actions = self.transition[state, :, :]
        is_available = np.sum(all_actions, 1)
        return np.arange(self.naction)[is_available > 0]

    def step(self, state, action):
        trans_proba = self.transition[state, action, :]
        assert np.sum(trans_proba) > 0
        next_state = np.random.choice(self.nstate, p=trans_proba)
        reward = self.reward[state, action]
        stop = self.is_terminal(next_state, action)
        
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

        self.terminal_state = self.matrix2lin(*terminal_state)
        self.init_transition()

        self.init_reward()
        


    def matrix2lin(self, coord1, coord2):
        return coord1 * self.length + coord2

    def lin2matrix(self, coord):
        coord1 = int(np.array(coord) / self.length)
        coord2 = coord % self.length
        return coord1, coord2

    def init_transition(self):0

if __name__ == "__main__":
    grid = GridWorld(10, 10, [0, 4])
    s = grid.reset()
    print(grid.lin2matrix(s))
    stop = False
    for i in range(1000):
        if stop:
            break
        action = np.random.choice(grid.available_actions(s))
        new_state, reward, stop = grid.step(s, action)
        print(grid.actions[action], grid.lin2matrix(new_state), reward)
        s = new_state
    
class Baird(Env):
    
    def __init__(self, epsilon):
        ''' Epsilon : probability of state six being terminal '''
        super(Baird, self).__init__()
        self.nstate = 6
        self.naction = 1
        self.epsilon = epsilon
        self.reward = np.zeros((self.nstate, self.naction))
        
        self.transition = np.zeros((self.nstate, self.naction, self.nstate))
        self.transition[:, :, -1] = 1
        
    def is_terminal(state, _=None):
        return np.random.rand(1) < self.epsilon