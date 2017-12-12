import numpy as np
from collections import namedtuple
import numbers
from tkinter import Tk
import tkinter.font as tkfont
import copy
from abc import ABC, abstractmethod
import gym

class Env(ABC):
    """Abstract environment class. All discrete environment should inherit from
    this class and define `nstate`, `naction`, `reward`, `transition` and
    the method `is_terminal`.
    """

    def __init__(self):
        self.nstate = None  # n
        self.naction = None  # a
        self.gamma = 1.

        self.reward = None  # n*a
        self.transition = None  # n*a*n
        self.state = None

    def reset(self, mu0=None):
        assert mu0 is None or len(mu0) == self.nstate
        self.state = np.random.choice(self.nstate, p=mu0)

    def available_actions(self):
        ''' Returns the indices of all available actions from state `state` '''
        all_actions = self.transition[self.state, :, :]
        is_available = np.sum(all_actions, 1)
        return np.arange(self.naction)[is_available > 0]

    def step(self, action):
        '''Warning: changes the internal value of state. If needed, this value
        should be stored before step is taken.'''
        if self.state is None:
            raise ValueError('The state should be initialized with env.step')
        trans_proba = self.transition[self.state, action, :]
        assert np.sum(trans_proba) > 0
        next_state = np.random.choice(self.nstate, p=trans_proba)
        reward = self.reward[self.state, action]
        stop = self.is_terminal(next_state, action)
        self.state = next_state
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

    def init_transition(self):
        self.transition = np.zeros((self.nstate, self.naction, self.nstate))
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

    def init_reward(self):
        ''' Start from the terminal state, apply all possible actions,
            add a reward to the inverse of each action '''

        self.reward = np.zeros((self.nstate, self.naction))
        x0, y0 = self.lin2matrix(self.terminal_state)

        deltax = [-1, 0, 1, 0]
        deltay = [0, -1, 0, 1]
        for a in range(self.naction):
                x1, y1 = x0 + deltax[a], y0 + deltay[a]
                if x1 >= 0 and x1 < self.height \
                    and y1 >= 0 and y1 < self.length:
                    j = self.matrix2lin(x1, y1)
                    self.reward[j, (a + 2) % 4] = 1  # Inverse action

    def is_terminal(self, state, _=None):
        return state == self.terminal_state


class Baird(Env):
    def __init__(self, epsilon=0.95, gamma=0.9999):
        ''' Epsilon : probability of state six being terminal
        Remark : we could stop the episode after only one iteration, since
        with Baird environment there is no update of the parameters
        when the state does not change.'''
        super(Baird, self).__init__()
        self.nstate = 6
        self.naction = 1
        self.epsilon = epsilon
        self.reward = np.zeros((self.nstate, self.naction))
        self.gamma = gamma

        self.transition = np.zeros((self.nstate, self.naction, self.nstate))
        self.transition[:, :, -1] = 1

    def is_terminal(self, state, _=None):
        return np.random.rand(1) < self.epsilon


class CartPole(Env):
    def __init__(self):
        ''' This class is an embedding of the CartPole-v0 environment
        from gym library.'''
        self.nstate = -1  # n
        self.naction = 2  # a: +1 or -1

        self.gym_env = gym.make('CartPole-v0')

    def reset(self, _=None):
        return self.gym_env.reset()

    def available_actions(self, state):
        ''' Returns the indices of all available actions from state `state`.
        '''
        return [0, 1]

    def step(self, action):
        return self.gym_env.step(action)

    def is_terminal(self, state, action):
        pass


if __name__ == "__main__":
    test = 'GridWorld'
    if test == 'GridWorld':
        grid = GridWorld(10, 10, [0, 4])
        grid.reset()
        print(grid.lin2matrix(grid.state))
        stop = False
        for i in range(1000):
            if stop:
                break
            action = np.random.choice(grid.available_actions())
            new_state, reward, stop = grid.step(action)
            print(grid.actions[action], grid.lin2matrix(new_state), reward)
    elif test == 'Baird':
        baird = Baird(0.5)
        baird.reset()
        print(baird.state)
        stop = False
        for i in range(1000):
            if stop:
                break
            action = np.random.choice(baird.available_actions())
            new_state, reward, stop = baird.step(action)
            print(baird.state)
