from minirl.policy.policy import Policy
import numpy as np
from minirl.rl_utils.parameters import Parameter


class TDPolicy(Policy):
    def __init__(self):
        self._approximator = None


    def set_approximator(self, approximator):
        self._approximator = approximator



class EpsilonGreedyPolicy(TDPolicy):
    def __init__(self, epsilon, decay = None, eps_min=None):
        self._epsilon = Parameter(value=epsilon,
                                  min_value=eps_min,
                                  decay=decay)


    def draw_action(self, state):
        if not np.random.uniform() < self._epsilon():
            q = self._approximator.predict(state)
            max_action = np.argwhere(q == np.max(q)).ravel()
            if len(max_action) > 1:
                max_action = np.array([np.random.choice(max_action)])

            return max_action[0]
        
        return np.array([np.random.choice(self._approximator.num_actions)])[0]
    

    def _update(self):
        self.epsilon = max(self._eps_min, self.epsilon * self._decay)

