
"""Main file : declares an environment and an algorithm, trains and plots results"""

import numpy as np
import numpy.random as npr
import numpy.linalg as LA
import envs.env as env
import algos.algo as algo
import models.model as model
import matplotlib.pyplot as plt


def plotTD0():
    plt.figure(1)
    plt.clf()
    plt.plot(average[0], label='TD0')
    plt.xlim([0, n_iter])
    plt.ylim([0, 1.05 * np.max(average[0])])
    plt.xlabel("Iteration")
    plt.ylabel("l2 distance to optimal parameters")
    plt.legend()
    plt.title("Baird's counterexample, averaged on {} experiments".format(
            n_experiments))
    plt.savefig('figures/bairdTD0.eps')
    plt.show()

def plotOthers():
    plt.figure(2)
    plt.clf()
    plt.plot(average[1], label='Residual gradients')
    plt.plot(average[2], label='Constrained gradients')
    plt.plot(average[3], label='GTD2')
    plt.plot(average[4], label='Residual constrained gradients')
    plt.xlim([0, n_iter])
    plt.ylim([0, 1.05 * np.max(average[1:])])
    plt.xlabel("Iteration")
    plt.ylabel("l2 distance to optimal parameters")
    plt.legend()
    plt.title("Baird's counterexample, averaged on {} experiments".format(
            n_experiments))
    plt.savefig('figures/bairdOthers.eps')
    plt.show()


if __name__ == '__main__':
#    npr.seed(0)

    # Define the environment, model and algorithm
    envi = env.Baird(epsilon=1, gamma=0.95)
    mod = model.LinearBaird()
    algorithms = [algo.TD0(envi, mod),
                  algo.ResidualGradient(envi, mod),
                  algo.ConstrainedGradient(envi, mod),
                  algo.GTD2(envi, mod),
                  algo.ConstrainedResidualGradient(envi, mod)]

    # Parameters
    alpha0 = 0.1
    n_iter = 1000
    n_experiments = 10

    hist = np.zeros((n_experiments, len(algorithms), n_iter))

    for i in range(n_experiments):
        # To initialize theta we follow Baird's 1995 recommendations:
        # All weights are positive, and the value of the terminal state
        # is much larger
        theta_init = np.abs(npr.randn(7))
        theta_init[-1] += 2
        for j, a in enumerate(algorithms):
            theta = np.copy(theta_init)
            envi.reset()
            for k in range(n_iter):
                if envi.stop:
                    envi.reset()
                hist[i, j, k] = LA.norm(mod.all_v(theta))
                s = envi.state
                action = a.policy()
                new_s, r, stop = envi.step(action)
                theta = a.update_parameters(s, new_s, r, theta, alpha0)

    average = np.mean(hist, axis=0)



    #    plotTD0()
    plotOthers()