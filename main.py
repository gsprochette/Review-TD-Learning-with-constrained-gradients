
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
envi = env.Baird(epsilon=0.95)
mod = model.LinearBaird()
alg1 = algo.TD0(envi, mod)
alg2 = algo.ResidualGradient(envi, mod)
alg3 = algo.ConstrainedGradient(envi, mod)

# Parameters
alpha = 1e-3
n_iter = 10000
n_experiments = 1

algorithmes = [alg1, alg2, alg3]
hist = np.zeros((n_experiments, len(algorithmes), n_iter))

for i in range(n_experiments):
    for j, a in enumerate(algorithmes):
        theta = 1e-2 * npr.randn(7)
        for k in range(n_iter):
            s = envi.reset()
            action = alg1.policy(s)
            new_s, r, stop = envi.step(s, action)
            theta = a.update_parameters(s, new_s, r, theta, alpha)

            hist[i, j, k] = LA.norm(theta)

average = np.mean(hist, axis=0)
hist0 = average[0]
hist1 = average[1]
hist2 = average[2]

# Display the results
plt.clf()
plt.plot(hist0, label='TD0')
plt.plot(hist1, label='Residual gradients')
plt.plot(hist2, label='Constrained gradients')
plt.xlabel("Iteration")
plt.ylabel("l2 norm of theta")
plt.legend()
plt.title("Baird's counterexample, averaged on {} experiments".format(
        n_experiments))
plt.show()
