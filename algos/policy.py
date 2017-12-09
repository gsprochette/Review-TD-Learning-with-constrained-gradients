

def epsilon_greedy(**kwargs):
    env = kwargs["env"]
    state = kwargs["state"]
    value_function = kwargs["V"]
    epsilon = kwargs["epsilon"]

    if env.localrandom.rand() < epsilon:  # exporation 
        return env.localrandom.choice(env.nstate)

    actions = env.available_actions(state)
    best_v, best_a = float("-inf"), None
    for a in actions:
        # problem : how de we get value function in next state ?