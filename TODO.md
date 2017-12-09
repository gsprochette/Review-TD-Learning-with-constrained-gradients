# Progress

### Milestones
Test Baird & gridworld
Find references about used environments
Look at other algorithms : Residual Gradients, Constraied RG, TD

1st part of report : corrections to be done on the paper
2nd part : reproducibility of presented experiments
3rd part : generalize to other environments ?

### Remarks about the paper
##### Uncertainty about the meaning of $g_v(s_{t+1})$ :
$g_v(s_{t+1})$ is defined as "$g_v(s_{t+1})$ is the gradient at $s_{t+1}$ that will change the value the most" (p.3), which is unclear for two reasons:
- the gradient of a quantity is already the direction that will change the value the most (and since the vector is renormalized in the next step, its norm does not matter)
- it is unclear what the "value" to be changed is.
We assume that it is defined as :
$$
  g_v(s_{t+1}) = \frac{\partial Q_\theta(s_{t+1}, a_{t+1}^*)}{\partial \theta} \quad \text{or} \quad
  g_v(s_{t+1}) = \frac{\partial V_\theta(s_{t+1})}{\partial \theta}
$$
Where $a^*$ is the action that realizes the maximum of $Q_\theta$ in $s^{t+1}$. Note that in all environments described, the action space is finite for each state. It is defined as the gradient of either $Q$ or $V$ depending on the context.

##### Correction on the definition of the projection used as constraint
The equations (6), (7), (8) and (9) are presented in a non-intuitive order, it should probably be (7), (8), (6), (9), and there seem to be an error in the equation (7):
Equation (8) is an orthonormal projection on the direction of $\hat g_v(s_{t+1})$, given that $\| \hat g_v(s_{t+1}) \| = 1$, which suggests that (7) should be $\hat g_v(s_{t+1}) = \frac{g_v(s_{t+1})}{\| g_v(s_{t+1}) \|}$.

##### Interpretation of the method
The proposed method tries to reduce the instability that appears when using a bootstrap. Its way of doing this is to project the update on the orthogonal of the "bootstrap update". 
- Let's first consider the common lookup-table case: if we apply the constrained method to Q-Learning with a finite number of state and Q being a simple table parameterized by $(\theta_{ij})_{i \in S, j \in A}$, we only update the Q function in the current state without updating the boostrap $\max_a Q_\theta (s_{t+1}, a)$. 

- This technique can be compared to fact that commonly, when evaluating the gradient of the TD error, we do not take into account the dependency of $\max_a Q_\theta (s_{t+1}, a)$ in $\theta$. What this paper claims is that constraining with this projection is more stable than ignoring the dependency.


### Other environment to be tested
Find other environments with weird TD behaviour (maybe in references from Baird)


### References
Baird : http://www.leemon.com/papers/1995b.pdf


### Reproducibility Challenge Constraints
