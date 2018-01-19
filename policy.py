import numpy as np
import torch
import torch.nn.functional as F


def softmax_(x):
    """Compute softmax values for each sets of scores in x."""
    try:
        e_x = np.exp(x - np.max(x))
    except FloatingPointError:
        print("x:\n{}".format(x))
        print("max x:\n{}".format(np.max(x)))
        raise

    res = e_x / e_x.sum()
    if np.sum(np.isnan(res)):
        print("x:\n{}".format(x))
        print("max x:\n{}".format(np.max(x)))
        print("e_x:\n{}".format(e_x))
        raise ValueError
    return res

# Policies for Q-Learning take as argument the Q-function

class ConstantAction(object):
    def __init__(self, idx):
        super(ConstantAction, self).__init__()
        self.idx = idx

    def __call__(self, qval, av_actions):
        return self.idx


def random_action(qval, av_actions):
    """Picks an available action uniformly at random."""

    action_idx = np.random.choice(av_actions)
    return int(action_idx)


def best_action(qval, av_actions):
    """Picks action that maximizes the Q-Function."""

    best_a = av_actions[np.argmax(qval[av_actions])]
    return int(best_a)


class EpsilonGreedyAction(object):
    def __init__(self, epsilon):
        super(EpsilonGreedyAction, self).__init__()
        self.eps = epsilon

    def __call__(self, qval, av_actions):
        if np.random.rand() > self.eps:
            return best_action(qval, av_actions)
        else:
            return random_action(qval, av_actions)


class EpsilonGreedyDecayAction(object):
    def __init__(self, initial_p):
        super(EpsilonGreedyDecayAction, self).__init__()
        self.eps = initial_p
        self.nsteps = 1

    def __call__(self, qval, av_actions):
        if np.random.rand() > self.eps / (1 + float(self.nsteps) / 3000):
            return best_action(qval, av_actions)
        else:
            return random_action(qval, av_actions)

    def step(self):
        self.nsteps += 1

    def reset(self):
        self.nsteps = 1


def softmax_action(qval, av_actions):
    """Picks an available action at random according to a distribution
    given by the softmax of the Q-Function.
    """

    probs = softmax_(qval[av_actions])
    try:
        action_idx = np.random.choice(av_actions, p=probs)
    except FloatingPointError:
        print("\tav_actions:\n{}".format(av_actions))
        print("\tprobs:\n{}".format(probs))
        raise
    return int(action_idx)


##############################################################
################ Target Values for Q-Learning ################
##############################################################

def best(qval):
    """Returns the best Q value in qval"""
    return torch.max(qval)

def softmax(qval):
    """Returns the expected value of the Q-Function under the
    Softmax policy.
    """
    probs = F.softmax(qval.squeeze(), 0)
    expected_qval = torch.sum(probs * qval)
    return expected_qval
