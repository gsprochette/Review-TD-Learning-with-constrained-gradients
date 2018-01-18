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
        self.state_ = None
        self.stop = False

    @property
    def state(self):
        """Used to modify the intern representation of states. For
        example, GridWorld needs to return the 2 coordinates
        corresponding to self.state_.
        """
        return self.state_

    def reset(self, mu0=None):
        assert mu0 is None or len(mu0) == self.nstate
        self.stop = False
        self.state_ = np.random.choice(self.nstate, p=mu0)

    def available_actions(self):
        ''' Returns the indices of all available actions from state `state` '''
        all_actions = self.transition[self.state_, :, :]
        is_available = np.sum(all_actions, 1)
        return np.arange(self.naction)[is_available > 0]

    def step(self, action):
        '''Warning: changes the internal value of state. If needed, this value
        should be stored before step is taken.'''
        if self.state_ is None:
            raise ValueError('The state should be initialized with env.step')
        trans_proba = self.transition[self.state_, action, :]
        assert np.sum(trans_proba) > 0
        next_state = np.random.choice(self.nstate, p=trans_proba)
        reward = self.reward[self.state_, action]
        stop = self.is_terminal(next_state, action)
        self.state_ = next_state
        self.stop = stop
        return next_state, reward, stop

    def is_terminal(self, state, action):
        pass


class GridWorld(Env):

    def __init__(self, gridh, gridl, terminal_state, gamma=0.95):
        super(GridWorld, self).__init__()
        self.height = gridh
        self.length = gridl
        self.nstate = gridh * gridl
        self.gamma = gamma

        self.actions = ['up', 'left', 'down', 'right']
        self.naction = 4

        self.terminal_state = self.matrix2lin(*terminal_state)
        self.transition = self.init_transition()

        self.init_reward()

    @property
    def state(self):
        return self.lin2matrix(self.state_)

    def matrix2lin(self, coord1, coord2):
        return coord1 * self.length + coord2

    def lin2matrix(self, coord):
        coord1 = int(np.array(coord) / self.length)
        coord2 = coord % self.length
        return coord1, coord2

    def init_transition(self):
        transition = np.zeros(
            (self.nstate, self.naction, self.nstate))
        deltax = [-1, 0, 1, 0]
        deltay = [0, -1, 0, 1]

        for i in range(self.nstate):
            x0, y0 = self.lin2matrix(i)
            for a in range(self.naction):
                x1, y1 = x0 + deltax[a], y0 + deltay[a]
                if x1 >= 0 and x1 < self.height \
                    and y1 >= 0 and y1 < self.length:
                    j = self.matrix2lin(x1, y1)
                    transition[i, a, j] = 1

        # for i in range(self.nstate):
        #     x0, y0 = self.lin2matrix(i)
        #     for a in range(self.naction):
        #         print("From {} ({}) taking action {}:\n{}\n".format(
        #             (x0, y0), i, a, transition[i, a, :]
        #         ))
        return transition

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

    def MSE(self, V_estimated):
        ''' Mean square error of the estimated state value function for the
        softmax policy.'''
        assert len(V_estimated) == len(self.V_softmax), \
            'len(V_estimated) should be 100'
        diff = self.V_softmax - V_estimated
        return np.dot(diff, diff)


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
        self.gamma = 0.999999

        self.gym_env = gym.make('CartPole-v0')
        self.state_ = self.reset()

    def reset(self, _=None):
        return self.gym_env.reset()

    def available_actions(self):
        ''' Returns the indices of all available actions from state `state`.
        '''
        return np.array([0, 1])

    def step(self, action):
        action = int(action)
        next_state, reward, stop, _ = self.gym_env.step(action)
        self.state_ = next_state
        self.stop = stop
        return (next_state, reward, stop)


class MountainCar(Env):
    def __init__(self):
        ''' This class is an embedding of the MountainCar-v0 environment
        from gym library. '''
        self.nstate = -1
        self.naction = 2
        self.gym_env = gym.make('MountainCar-v0')
        self.gamma = 1
        self.state_ = None
        self.terminated = False

    def reset(self, _=None):
        self.terminated = False
        self.state_ = self.gym_env.reset()

    def available_actions(self, _=None):
        ''' 0: push left; 1: no push ; 2: push right.'''
        return np.array([0, 1, 2])

    def step(self, aciton):
        next_s, reward, self.terminated, info = self.gym_env.step(action)
        self.state_ = next_s
        return next_s, reward, self.terminated

    def is_terminal(self, state, action):
        return self.terminated


if __name__ == "__main__":
    test = 'GridWorld'
    if test == 'GridWorld':
        grid = GridWorld(10, 10, [0, 4])
        grid.reset()
        print(grid.lin2matrix(grid.state))
        stop = False
        test_step = False
        if test_step:
            for i in range(1000):
                if stop:
                    break
                action = np.random.choice(grid.available_actions())
                new_state, reward, stop = grid.step(action)
                print(grid.actions[action], grid.lin2matrix(new_state), reward)
        test_softmax_evaluation = False
        if test_softmax_evaluation:
            v = grid.softmax_evaluation()
            print(v)
        V = np.zeros(100)
        mse = grid.MSE(V)
        print('Mean square error', mse)
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
    elif test == 'MountainCar':
        mc = MountainCar()
        mc.reset()
        print(mc.state)
        stop = False
        for i in range(1000):
            if stop:
                break
            action = np.random.choice(mc.available_actions())
            next_s, reward, stop = mc.step(action)
            print(mc.state)
    else:
        print('Environment not implemented')
