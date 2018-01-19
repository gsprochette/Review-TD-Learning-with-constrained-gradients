import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
import matplotlib.pyplot as plt
import envs.env_torch as env
import models.model_torch as model
import algos.algo_torch as algo
import policy

def param_norm(params):
    norms = [torch.norm(param, p=2).data[0] ** 2 for param in params]
    norm = np.sqrt(np.sum(norms))
    return norm

def episode(algo, max_iter=200, param_init=None):
    """Trains on one full episode"""
    algo.env.reset(algo.mu0)
    state = algo.env.state
    stop = False
    reward_acc = []
    param_dist = []
    if hasattr(state, '__iter__'):  # list or tuple
        state = Variable(torch.Tensor(state))
        state = state.unsqueeze(0).unsqueeze(1)
    else:
        state = Variable(torch.Tensor([state]))

    iter = 0
    while not stop and iter < max_iter:
        iter += 1
        # take action
        action_idx = algo.policy()
        # print("state: {}".format(state))
        # print("action idx: {}".format(action_idx))
        next_state, reward, stop = algo.env.step(action_idx)
        old_state = state
        state = algo.env.state
        if hasattr(state, '__iter__'):  # list or tuple
            state = Variable(torch.Tensor(state))
            state = state.unsqueeze(0).unsqueeze(1)
        else:
            state = Variable(torch.Tensor([state]))

        # update model parameters
        algo.update(old_state, state, reward, action_idx)

        # log
        reward_acc.append(reward)

        if param_init is not None:
            param_var = [param - param_init[i]
                         for i, param in enumerate(algo.model.parameters())]
            param_dist.append(param_norm(param_var))
    algo.nepisode += 1
    algo.rewards.append(reward_acc)
    return reward_acc, param_dist


if __name__ == "__main__":
    # seed randomness ?
    np.seterr(invalid='raise')  # sauf underflow error...

    # env_func = lambda: env.GridWorld(10, 10, (0, 0))
    env_func = lambda: env.CartPole()
    mod = lambda model0: deepcopy(model0)
    pol = policy.EpsilonGreedyDecayAction(0.05)

    alpha0 = 1e-3
    # alpha = lambda episode: alpha0 / (1 + episode / T0)
    alpha = lambda episode: alpha0
    args = lambda model0: (env_func(), mod(model0), pol)
    kwargs = lambda constr, res: dict(
        lr_fun=alpha, target="best", constraint=constr, residual=res)

    TD0 = lambda model0: algo.TD0(
        *args(model0), **kwargs(False, False))
    TD0c = lambda model0: algo.TD0(
        *args(model0), **kwargs(True, False))
    RTD0 = lambda model0: algo.TD0(
        *args(model0), **kwargs(False, True))
    RTD0c = lambda model0: algo.TD0(
        *args(model0), **kwargs(True, True))

    QL = lambda model0: algo.QLearning(
        *args(model0), **kwargs(False, False))
    QLc = lambda model0: algo.QLearning(
        *args(model0), **kwargs(True, False))
    RQL = lambda model0: algo.QLearning(
        *args(model0), **kwargs(False, True))
    RQLc = lambda model0: algo.QLearning(
        *args(model0), **kwargs(True, True))

    DQN = lambda model0: algo.DeepQLearning(
        *args(model0), **kwargs(False, False))
    DQNc = lambda model0: algo.DeepQLearning(
        *args(model0), **kwargs(True, False))
    RDQN = lambda model0: algo.DeepQLearning(
        *args(model0), **kwargs(False, True))
    RDQNc = lambda model0: algo.DeepQLearning(
        *args(model0), **kwargs(True, True))

    algorithms = [
        TD0, TD0c, RTD0, RTD0c,
        QL, QLc, RQL, RQLc,
        DQN, DQNc, RDQN, RDQNc
        ]
    algorithms = [
        QL,
        ]
    n_algo = len(algorithms)

    nexperiment = 1
    nepisode = 15000
    hist = np.zeros((n_algo, nepisode))
    param_variation = [0.]
    for iexp in range(nexperiment):
        print("experiment {}".format(iexp))
        # same initialization for all algorithms
        model0 = model.CartpoleNet()
        param0 = list(model0.parameters())
        algos = [alg(model0) for alg in algorithms]

        for i, alg in enumerate(algos):
            print(alg.name)
            try:
                pol.reset()
            except AttributeError:
                pass
            for iepisode in range(nepisode):
                alg.scheduler.step()  # update learning rate
                rewards, param_dist = episode(alg, param_init=param0)
                # train for one episode

                # rewards histogram
                hist[i, iepisode] += float(np.sum(rewards)) / nexperiment
                param_variation = param_variation + param_dist
        print('\n')

    # plot results
    plt.clf()
    for i in range(n_algo):
        plt.plot(hist[i, :], **algos[i].plot_kwargs())
    plt.xlim([0, nepisode])
    plt.ylim([0, 20])
    plt.xlabel("Iteration")
    plt.ylabel("Cumulated Reward")
    plt.legend()
    plt.title(
        "Learning on Cartpole" \
        + ("" if nexperiment == 1 else \
        ", averaged on {} experiments".format(nexperiment)))
    plt.savefig('Cartpole')
    plt.figure()
    plt.plot(param_variation)
    plt.xlabel("Number of parameter updates")
    plt.ylabel("Distance to initial parameters")
    plt.show()
