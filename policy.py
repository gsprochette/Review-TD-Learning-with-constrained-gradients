import numpy as np
import torch
import torch.nn.functional as F


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Policies for Q-Learning take as argument the Q-function

class ConstantAction(object):
    def __init__(self, idx):
        super(ConstantAction, self).__init__()
        self.idx = idx

    def __call__(self, qval):
        return self.idx


def random_action(qval):
    """Picks an available action uniformly at random."""
    return np.random.choice(torch.numel(qval))


def best_action(qval):
    """Picks action that maximizes the Q-Function."""

    best_a = np.argmax(qval)
    return best_a


def softmax_action(qval):
    """Picks an available action at random according to a distribution
    given by the softmax of the Q-Function.
    """

    probs = softmax(qval)
    action = np.choice(np.size(qval, p=probs))
    return action


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
    probs = F.softmax(qval)
    expected_qval = torch.sum(probs * qval)
    return expected_qval