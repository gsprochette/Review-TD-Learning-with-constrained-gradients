# Review : TD Learning with Constrained Gradients

This repository is a reviews of the submission <a href="https://openreview.net/pdf?id=Bk-ofQZRb">TD Learning with Constrained Gradients</a>. The gradient projection was implemented using <a href="http://pytorch.org/">Pytorch</a>'s autograd mechanics. Each file either has a `*_torch` equivalent or is compatible with the torch implementation. 

### Quick test

To install the gym package:
`pip install --user gym`

In order to run the baseline, you must first install tensorflow (https://www.tensorflow.org/install/) and activate the corresponding environment: 
The openai baseline library must also be installed (https://github.com/openai/baselines)

Projection on the Baird environment can be tested using:
`python main_baird.py`
and the GridWorld environment can be tested using:
`python main_batch.py`

### Code Structure

##### `main*.py`
Main file : chooses an environment and an algorithm, trains and plots results. In particular, `main_batch.py` creates batchs of transition in order to learn in a more stable way.

##### `envs/`
Defines the environments corresponding to different problems. Each environment contains one method `reset` that returns an initial state, and one method `step` that takes as argument the current state and returns the next state, a rewart and a boolean that indicates whether or not the episode is over.

##### `algos/`
Defines the algorithms that learn to optimize the rewards on an environment. One main class `Algo` defines basic methods and a method `episode` that iteratively calls `env.step` and relies on the methods `action` and `update_parameters` to train. All other algorithms inherit from this main class and should redefine the methods `action` and `update_parameters`.

##### `models/`
Defines the models used for approximating either the state value function or the action value function. Each model has a method `g_v` which computes the gradient of said function with regards to the state, as described in the paper.

##### `policy.py`
Defines the different policies used for learning. The main policies are random, greedy and softmax. Greedy policy chooses the best action from the current approximated action value function. Softmax policy randomly picks an action with probabilies computed as the softmax of the current approximated action value function. Those last two have exploration version of themselves : epsilon-greedy and epsilon-softmax, which choose a random action with probability epsilon.

##### `obj/`
Directory used for storing action/state value functions learnt
