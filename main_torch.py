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

def episode(algo):
    """Trains on one full episode"""
    algo.env.reset(algo.mu0)
    state = algo.env.state
    stop = False
    reward_acc = []
    if hasattr(state, '__iter__'):  # list or tuple
        state = Variable(torch.Tensor(state))
        state = state.unsqueeze(0).unsqueeze(1)
    else:
        state = Variable(torch.Tensor([state]))
    while not stop:
        # take action
        action_idx = algo.policy()  # WARNING: this should choose an AVAILABLE action
        print(state)
        print(action_idx)
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
    algo.nepisode += 1
    algo.rewards.append(reward_acc)


if __name__ == "__main__":
    # seed randomness ?

    env_func = lambda: env.GridWorld(3, 3, (0, 0))
    mod = lambda model0: deepcopy(model0)
    pol = policy.best_action

    alpha0, T0 = 0.1, 500
    # alpha = lambda episode: alpha0 / (1 + episode / T0)
    alpha = lambda episode: alpha0
    args = lambda model0: (env_func(), mod(model0), pol)
    kwargs = lambda constr, res: dict(
        lr_fun=alpha, target='best', constraint=constr, residual=res)

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
        QL, QLc, RQL, RQLc,
        ]
    n_algo = len(algorithms)

    nexperiment = 1
    nepisode = 200
    hist = np.zeros((n_algo, nepisode))
    for iexp in range(nexperiment):
        # same initialization for all algorithms
        model0 = model.Net()
        algos = [algo(model0) for algo in algorithms]

        for iepisode in range(nepisode):
            for i, algo in enumerate(algos):
                algo.scheduler.step()  # update learning rate
                episode(algo)  # train for one episode

                value = algo.model.all_v()
                hist[i, iepisode] += torch.norm(value, p=2)
    hist /= nexperiment

    # plot results
    plt.clf()
    for i in range(n_algo):
        plt.plot(hist[i, :], **algos[i].plot_kwargs())
    plt.xlim([0, nepisode])
    plt.ylim([0, 1.3 * np.max(hist[:, 0])])
    plt.xlabel("Iteration")
    plt.ylabel("l2 norm of theta")
    plt.legend()
    plt.title(
        "Learning on GridWorld" \
        + ("" if nexperiment == 1 else \
        ", averaged on {} experiments".format(nexperiment)))
    plt.show()
