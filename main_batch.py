import numpy as np
from copy import deepcopy
import itertools
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
from replay_buffer import ReplayBuffer

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


def train(algo_func, model0s,
          max_iter=200, batch_size=32, buffer_size=50000):
    buffer = ReplayBuffer(buffer_size)

    # create dataset
    for model0 in model0s:  # iterates on all experiments
        alg = algo_func(model0)

        episode_rewards = [0.0]
        alg.env.reset(alg.mu0)
        state = alg.env.state
        for t in itertools.count():
            action_idx = alg.policy()
            new_state, reward, done = alg.env.step(action_idx)

            # store transition
            buffer.add(state, action_idx, reward, new_state, float(done))
            state = new_state

            episode_rewards[-1] += reward
            
            if done and t > 100:
                print(episode_rewards[-1], np.mean(episode_rewards[-101:-1]),
                      alg.pol.epsilon())
            
            if done:
                alg.env.reset(alg.mu0)
                state = alg.env.state
                episode_rewards.append(0.0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                pass
                # Show off the result
                #env.render()
                #continue
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 100: # 0:
                    states, actions, rewards, nstates, _ = buffer.sample(32)
                    
                    states = Variable(torch.FloatTensor(states))
                    nstates = Variable(torch.FloatTensor(nstates))

                    alg.batch_update(states, nstates, np.ones_like(rewards), actions)


if __name__ == "__main__":
    # seed randomness ?
    np.seterr(invalid='raise')  # sauf underflow error...

    # env_func = lambda: env.GridWorld(10, 10, (0, 0))
    env_func = lambda: env.CartPole()
    mod = lambda model0: deepcopy(model0)
    pol = policy.EpsilonGreedyDecayAction(10000, 1.0, 0.02)

    alpha0, T0 = 1e-3, 200
    alpha = lambda episode: alpha0 / (1 + episode / T0)
    # alpha = lambda episode: alpha0
    args = lambda model0: (env_func(), mod(model0), pol)
    kwargs = lambda constr, res: dict(
        lr_fun=alpha, target="softmax", constraint=constr, residual=res)

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
        DQNc,
        ]
    n_algo = len(algorithms)

    nexperiment = 1
    nepisode = 4000
    hist = np.zeros((n_algo, nepisode))
    param_variation = [0.]

    model0s = [model.CartpoleNet() for _ in range(nexperiment)]
    for algo_func in algorithms:
        train(algo_func, model0s)

    # plot results
    plt.clf()
    for i in range(n_algo):
        plt.plot(hist[i, :], **algos[i].plot_kwargs())
    plt.xlim([0, nepisode])
    plt.ylim([0, 220])
    plt.xlabel("Iteration")
    plt.ylabel("Cumulated Reward")
    plt.legend()
    plt.title(
        "Batch Learning on Cartpole" \
        + ("" if nexperiment == 1 else \
        ", averaged on {} experiments".format(nexperiment)))
    plt.savefig('Cartpole')
    plt.figure()
    plt.plot(param_variation)
    plt.xlabel("Number of parameter updates")
    plt.ylabel("Distance to initial parameters")
    plt.show()
