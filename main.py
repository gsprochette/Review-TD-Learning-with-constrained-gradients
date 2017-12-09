
"""Main file : declares an environment and an algorithm, trains and plots results"""

import numpy as np
import numpy.random as npr
import envs.env as env
import matplotlib.pyplot as plt

# npr.seed(0)

# Baird
baird = env.Baird(epsilon=0.8)
theta = 1e-2 * npr.randn(7)

M = np.zeros((6, 7))
# Each line i : the parameters associated to state i
M[:, 0] = 1
for i in range(5):
    M[i, i + 1] = 2
M[5, 0] = 2
M[5, 6] = 1


def V(theta):
    return M @ theta

# Q-learning : delta = V(s) - V_s(new)
alpha = 1e-1
n_iter = 100
hist = np.zeros(n_iter)
for i in range(n_iter):
    s = baird.reset()
    stop = False
    for j in range(100):
        action = np.random.choice(baird.available_actions(s))
        new_state, reward, stop = baird.step(s, action)
        theta += 2 * alpha * (M[s] @ theta - M[new_state] @ theta) * (M[s])
        #print(theta)
        #print((M[s] @ theta - M[new_state] @ theta))
        s = new_state
        if stop:
            break
    hist[i]  = np.sqrt(np.dot(theta, theta))

plt.clf()
plt.plot(hist[50:])
print(theta)
print(np.dot(theta, theta))