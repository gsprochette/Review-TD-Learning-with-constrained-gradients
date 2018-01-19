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


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)

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
    def __init__(self, schedule_timesteps, initial_p, final_p):
        super(EpsilonGreedyDecayAction, self).__init__()
        self.schedule = LinearSchedule(
            schedule_timesteps, initial_p=initial_p, final_p=final_p)
        self.nsteps = 0

    def __call__(self, qval, av_actions):
        if np.random.rand() > self.epsilon():
            return best_action(qval, av_actions)
        else:
            return random_action(qval, av_actions)

    def epsilon(self):
        return self.schedule.value(self.nsteps)

    def step(self):
        self.nsteps += 1

    def reset(self):
        self.nsteps = 0


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


class EpsilonSoftmaxAction(object):
    def __init__(self, schedule_timesteps, initial_p, final_p):
        super(EpsilonSoftmaxAction, self).__init__()
        self.schedule = LinearSchedule(
            schedule_timesteps, initial_p=initial_p, final_p=final_p)
        self.nsteps = 0

    def __call__(self, qval, av_actions):
        if np.random.rand() > self.epsilon():
            return softmax_action(qval, av_actions)
        else:
            return random_action(qval, av_actions)

    def epsilon(self):
        return self.schedule.value(self.nsteps)

    def step(self):
        self.nsteps += 1

    def reset(self):
        self.nsteps = 0


##############################################################
################ Target Values for Q-Learning ################
##############################################################

def best(qval, dim=0):
    """Returns the best Q value in qval"""
    expected_qval, _ = torch.max(qval, dim)
    return expected_qval

def softmax(qval, dim=0):
    """Returns the expected value of the Q-Function under the
    Softmax policy.
    """
    probs = F.softmax(qval.squeeze(), dim)
    expected_qval = torch.sum(probs * qval, dim)
    return expected_qval
