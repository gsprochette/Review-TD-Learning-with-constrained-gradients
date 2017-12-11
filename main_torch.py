import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
import matplotlib.pyplot as plt
import envs.env as Env
import models.model_torch as Model
import algos.algo_torch as Algo


def episode(algo, optimizer):  # Should be in main.py
    """Trains on one full episode"""
    algo.env.reset(algo.mu0)
    state = algo.env.state
    stop = False
    reward_acc = []
    while not stop:
        # take action
        action = algo.policy()
        reward, stop = algo.env.step(action)
        old_state = state
        state = algo.env.state

        # update model parameters
        algo.update(old_state, state, reward, optimizer)

        # log
        reward_acc.append(reward)
    algo.nepisode += 1
    algo.rewards.append(reward_acc)


def update_parameters(parameters, g_update, optimizer):
    for param, grad in zip(parameters, g_update):
        param.grad = grad
    optimizer.step()


if __name__ == "__main__":
    # seed randomness ?

    env = Env.Baird
    model = Model.LinearBaird
    policy = lambda _: 0  # Baird : only one action possible

    RG = Algo.ResidualGradient(
        env(), model(), policy, constraint=False)
    RGc = Algo.ResidualGradient(
        env(), model(), policy, constraint=True)
    algorithms = [RG, RGc]

    alpha0, T0 = 0.1, 100
    alpha = lambda episode: alpha0 / (1 + episode / T0)

    optimizers = [optim.SGD(algo.model.parameters(), lr=1.)
                  for algo in algorithms]
    schedulers = [
        optim.lr_scheduler.LambdaLR(opt, lr_lambda=alpha)
        for opt in optimizers
    ]

    nexperiment = 1
    nepisode = 100
    hist = np.zeros((len(algorithms), nepisode))
    for iexp in range(nexperiment):
        for iepisode in range(nepisode):
            for i, algo in enumerate(algorithms):
                episode(algo, schedulers[i])
                hist[i, iepisode] += torch.norm(algo.model.theta, p=2)
    hist /= nexperiment
    print(hist)

    # plot results
    plt.clf()
    plt.plot(hist[0, :], label='Residual Gradient -- Unconstrained')
    plt.plot(hist[1, :], label='Residual Gradient -- Constrained')
    plt.xlim([0, nepisode])
    plt.ylim([0, 1.05 * np.max(hist)])
    plt.xlabel("Iteration")
    plt.ylabel("l2 norm of theta")
    plt.legend()
    plt.title(
        "Baird's counterexample" \
        + "" if nexperiment == 1 else \
        ", averaged on {} experiments".format(nexperiment))
    plt.show()