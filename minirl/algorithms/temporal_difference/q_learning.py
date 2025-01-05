import numpy as np

from minirl.algorithms.temporal_difference.temporal_difference import TD

class QLearning(TD):
    def __init__(self, env_info, policy, learning_rate):

        approximator = ...

        super().__init__(env_info, policy, approximator, learning_rate)


    def _update(self, state, action, reward, next_state, done):
        q_current = self.approximator[state, action]

        q_next = np.max(self.approximator[next_state, :]) if not done else 0.

        self.approximator[state, action] = q_current + self._alpha(state, action) * (
            reward + self.env_info.gamma * q_next - q_current
        )