# Review : TD Learning with Constrained Gradients

This repository is a reviews of the paper submission <a href="https://openreview.net/pdf?id=Bk-ofQZRb">TD Learning with Constrained Gradients</a>.

### Quick test

### Code Structure

##### envs/
Defines the environments corresponding to different problems. Each environment contains one method `reset` that returns an initial state, and one method `step` that takes as argument the current state and returns the next state, a rewart and a boolean that indicates whether or not the episode is over.

##### algos/
Defines the algorithms that learn to optimize the rewards on an environment. One main class `Algo` defines basic methods and a method `episode` that iteratively calls `env.step` and relies on the methods `action` and `update_parameters` to train. All other algorithms inherit from this main class and should redefine the methods `action` and `update_parameters`.

##### main.py
Main file : chooses an environment and an algorithm, trains and plots results.
