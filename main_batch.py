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
from utils import save_obj
import string

def save_qv_func(alg, alg_name):
    height, length = alg.env.height, alg.env.length
    q_func = np.zeros((height, length, 4))
    v_func = np.zeros((height, length))
    for i in range(height):
        for j in range(height):
            state = Variable(torch.FloatTensor([i, j]))
            state = state.unsqueeze(0).unsqueeze(1)
            q_state = alg.model(state).squeeze(0).squeeze(0)
            v_state = alg.target(q_state)

            v_func[i, j] = v_state.data[0]
            q_func[i, j, :] = q_state.data.numpy()

    qv_func = dict(q=q_func, v=v_func)
    rand_str = ''.join(np.random.choice(list(string.ascii_uppercase), 5))
    name = "qvfunc_{}_{}".format(alg_name, rand_str)
    save_obj(qv_func, name)
    print("saved as {}".format(name))


def train(algo_func, model0s, alg_name,
          batch_size=32, buffer_size=50000):
    buffer = ReplayBuffer(buffer_size)

    exp_times = []
    # create dataset
    for model0 in model0s:  # iterates on all experiments
        alg = algo_func(model0)

        # episode_rewards = [0.0]
        episode_times = [0.0]
        alg.env.reset(alg.mu0)
        state = alg.env.state
        iter = 0
        for t in itertools.count():
            iter += 1
            action_idx = alg.policy()
            new_state, reward, done = alg.env.step(action_idx)

            # store transition
            buffer.add(state, action_idx, reward, new_state, float(done))
            state = new_state

            # episode_rewards[-1] += reward
            episode_times[-1] += 1.0

            if done or iter >= 300:
                alg.env.reset(alg.mu0)
                state = alg.env.state
                iter = 0
                # print(episode_times[-1])
                # episode_rewards.append(0.0)
                episode_times.append(0.0)

            # is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            is_solved = t > 5000 and np.mean(episode_times[-101:-1]) <= 20
            if t > 1000 and (t+1) % 1000 == 0:
                print(np.mean(episode_times[-51:-1]))
            if is_solved:
                break
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 200: # 0:
                    # states, actions, rewards, nstates, _ = buffer.sample(batch_size)
                    states, actions, rewards, nstates, _ = buffer.sample(batch_size)

                    states = Variable(torch.FloatTensor(states))
                    nstates = Variable(torch.FloatTensor(nstates))

                    # alg_nonbatch = deepcopy(alg)
                    # param0 = [param.data.numpy() for param in alg_nonbatch.model.parameters()]
                    # alg_nonbatch.update(states[0], nstates[0], 1.0, int(actions[0]))
                    # grad0 = [param.grad.data.numpy() for param in alg_nonbatch.model.parameters()]

                    # param1 = [param.data.numpy() for param in alg.model.parameters()]
                    alg.batch_update(states, nstates, np.ones_like(rewards), actions)
                    # grad1 = [param.grad.data.numpy() for param in alg.model.parameters()]

                    # print(np.shape(param0), np.shape(param1))
                    # print(np.shape(grad0), np.shape(grad1))
                    # print([np.linalg.norm(p0 - p1) for p0, p1 in zip(param0, param1)])
                    # print([np.linalg.norm(g0 - g1) for g0, g1 in zip(grad0, grad1)])
                    # alg.scheduler.step()
        exp_times.append(episode_times)
        save_qv_func(alg, alg_name)
    return exp_times


if __name__ == "__main__":
    # seed randomness ?
    np.seterr(invalid='raise')  # sauf underflow error...
    seed = np.random.randint(1, 2 ** 20)
    print("seed: {}".format(seed))
    torch.manual_seed(seed)

    env_func = lambda: env.GridWorld(10, 10, (0, 0))
    # env_func = lambda: env.CartPole()
    mod = lambda model0: deepcopy(model0)
    pol = policy.EpsilonGreedyDecayAction(10000, 1.0, 0.02)
    #pol = policy.EpsilonSoftmaxAction(30000, 1.0, 0.01)

    alpha0, T0 = 1e-3, 2000
#    alpha = lambda episode: alpha0 / (1 + episode / T0)
    alpha = lambda episode: alpha0
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
        RDQN,
        ]
    n_algo = len(algorithms)

    nexperiment = 1
    model0s = [model.GridNet() for _ in range(nexperiment)]
    # model0s = [model.CartpoleNet() for _ in range(nexperiment)]

    for algo_func in algorithms:
        hist = train(algo_func, model0s, "RDQN", batch_size=32)

    print("seed: {}".format(seed))

    # plot results
    plt.clf()
    for i in range(nexperiment):
        plt.plot(hist[i], **algorithms[i](model0s[0]).plot_kwargs())
    plt.ylim([0, 320])
    plt.xlabel("Episodes")
    plt.ylabel("Time spent before terminal state")
    plt.legend()
    plt.title("Batch Learning on GridWorld")
    plt.savefig('GridWorld')

    # smooth results for visibility
    psmooth = 20
    plt.figure()
    for i in range(nexperiment):
        n = len(hist[i])
        smooth_hist = [np.mean(hist[i][k:k+psmooth]) for k in range(n - psmooth)]
        plt.plot(range(1, n - psmooth + 1), smooth_hist, **algorithms[i](model0s[0]).plot_kwargs())
    plt.ylim([0, 320])
    plt.xlabel("Episodes")
    plt.ylabel("Time spent before terminal state")
    plt.legend()
    plt.title("Batch Learning on GridWorld")
    plt.savefig('GridWorld')
    plt.show()