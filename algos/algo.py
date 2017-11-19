
"""Defines the class `Algo` : all algorithm learning at each step
should inherit from `Algo`.
"""


class Algo:
    """Defines the methods common to all training algorithm
    defined in this directory.
    """

    def __init__(self, env, mu0):
        super(Algo, self).__init__()

        self.env = env  # environment
        self.mu0 = mu0  # initial distribution
        self.parameters = None  # to be defined in class instances

        # initialize training informations
        self.nepisode = 0
        self.rewards = []


    def episode(self):
        """Trains on one full episode"""

        state = self.env.reset(self.mu0)
        reward_acc = 0
        stop = False
        while not stop:
            new_state, reward, stop = self.env.step()
            self.update_parameters(state, new_state, reward)
            reward_acc += reward
            state = new_state
        self.nepisode += 1
        self.rewards.append(reward_acc)


    def action(self, state):
        """Decides what action to take at state `state`.
        To be defined in class instances.
        """
        pass


    def update_parameters(self, state, new_state, reward):
        """Updates the parameters according to the last step.
        To be defined in class instances.
        """
        pass
