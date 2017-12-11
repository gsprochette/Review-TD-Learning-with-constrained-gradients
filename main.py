
"""Main file : declares an environment and an algorithm, trains and plots results"""

import numpy as np
import numpy.random as npr
import numpy.linalg as LA
import envs.env as env
import algos.algo as algo
import models.model as model
import matplotlib.pyplot as plt

# npr.seed(0)


# Define the environment, model and algorithm
envi = env.Baird(epsilon=0.8, gamma=0.95)
mod = model.LinearBaird()
alg0 = algo.TD0(envi, mod)
alg1 = algo.ResidualGradient(envi, mod)
alg2 = algo.ConstrainedGradient(envi, mod)
alg3 = algo.GTD2(envi, mod)
alg4 = algo.ConstrainedResidualGradient(envi, mod)
# Parameters
alpha0 = 0.1
n_iter = 2000
n_experiments = 1

algorithms = [alg0, alg1, alg2, alg3, alg4]
hist = np.zeros((n_experiments, len(algorithms), n_iter))

def alpha(alpha0, j, T0=100):
    return alpha0 / (1 + j / T0)

for i in range(n_experiments):
    theta_init = npr.randn(7)
    for j, a in enumerate(algorithms):
        theta = np.copy(theta_init)
        stop = False
        s = envi.reset()
        for k in range(n_iter):
            if stop:
                stop = False
                s = envi.reset()

            hist[i, j, k] = LA.norm(theta)
            action = alg1.policy(s)
            new_s, r, stop = envi.step(s, action)
            theta = a.update_parameters(s, new_s, r, theta, alpha0)
            s = new_s
average = np.mean(hist, axis=0)
#print(theta)
# Display the results
plt.clf()
#plt.plot(average[0], label='TD0')
plt.plot(average[1], label='Residual gradients')
plt.plot(average[2], label='Constrained gradients')
plt.plot(average[3], label='GTD2')
plt.plot(average[4], label='Residual constrained gradients')
plt.xlim([0, n_iter])
plt.ylim([0, 1.05 * np.max(average[1:])])
plt.xlabel("Iteration")
plt.ylabel("l2 norm of theta")
plt.legend()
plt.title("Baird's counterexample, averaged on {} experiments".format(
        n_experiments))
plt.show()
