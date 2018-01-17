import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import torch.optim as optim
import matplotlib.pyplot as plt
import envs.env as env
import models.model_torch as model
import algos.algo_torch as algo
import policy


def episode(algo):
    """Trains on one full episode"""
    algo.env.reset(algo.mu0)
    state = algo.env.state
    stop = False
    reward_acc = []
    while not stop:
        # take action
        action_idx = algo.policy()
        next_state, reward, stop = algo.env.step(action_idx)
        old_state = state
        state = algo.env.state

        # update model parameters
        algo.update(old_state, state, reward, action_idx)

        # log
        reward_acc.append(reward)
    algo.nepisode += 1
    algo.rewards.append(reward_acc)


if __name__ == "__main__":
    # seed randomness ?

    env_func = lambda: env.Baird(epsilon=0.8, gamma=0.95)
    mod = model.LinearBaird
    policy = lambda _: 0  # Baird : only one action possible

    alpha0, T0 = 0.1, 500
    # alpha = lambda episode: alpha0 / (1 + episode / T0)
    alpha = lambda episode: alpha0
    args = lambda theta0: (env_func(), mod(theta0), policy)
    kwargs = lambda constr, res: dict(
        lr_fun=alpha, target='best', constraint=constr, residual=res)

    TD0 = lambda theta0: algo.TD0(
        *args(theta0), **kwargs(False, False))
    TD0c = lambda theta0: algo.TD0(
        *args(theta0), **kwargs(True, False))
    RTD0 = lambda theta0: algo.TD0(
        *args(theta0), **kwargs(False, True))
    RTD0c = lambda theta0: algo.TD0(
        *args(theta0), **kwargs(True, True))

    QL = lambda theta0: algo.QLearning(
        *args(theta0), **kwargs(False, False))
    QLc = lambda theta0: algo.QLearning(
        *args(theta0), **kwargs(True, False))
    RQL = lambda theta0: algo.QLearning(
        *args(theta0), **kwargs(False, True))
    RQLc = lambda theta0: algo.QLearning(
        *args(theta0), **kwargs(True, True))

    DQN = lambda theta0: algo.DeepQLearning(
        *args(theta0), **kwargs(False, False))
    DQNc = lambda theta0: algo.DeepQLearning(
        *args(theta0), **kwargs(True, False))
    RDQN = lambda theta0: algo.DeepQLearning(
        *args(theta0), **kwargs(False, True))
    RDQNc = lambda theta0: algo.DeepQLearning(
        *args(theta0), **kwargs(True, True))

    algorithms = [
        TD0, TD0c, RTD0, RTD0c,
        QL, QLc, RQL, RQLc,
        DQN, DQNc, RDQN, RDQNc
        ]
    n_algo = len(algorithms)

    nexperiment = 1
    nepisode = 200
    hist = np.zeros((n_algo, nepisode))
    for iexp in range(nexperiment):
        # same initialization for all algorithms
        theta0 = mod().init_theta()
        algos = [algo(theta0) for algo in algorithms]

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
        "Baird's counterexample" \
        + ("" if nexperiment == 1 else \
        ", averaged on {} experiments".format(nexperiment)))
    plt.show()
