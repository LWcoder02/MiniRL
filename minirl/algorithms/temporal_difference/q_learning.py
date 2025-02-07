import numpy as np

from minirl.algorithms.temporal_difference.temporal_difference import TD
from minirl.approximators.table import Table

class QLearning(TD):
    def __init__(self, environment, policy, learning_rate):

        env_info = environment.get_environment_info()

        approximator = Table(env_info.num_actions)

        super().__init__(environment, policy, approximator, learning_rate)


    def _update_experimental(self, state, action, reward, next_state, done):
        q_current = self.approximator.get(state, action)

        q_next = np.max(self.approximator.get(next_state)) if not done else 0.

        update_value = q_current + self._alpha(state, action) * (
            reward + self.env_info.gamma * q_next - q_current
        )

        self.approximator.set(update_value, state, action)


    def _update(self, state, action, reward, next_state, terminal):
        q_current = self.approximator.get(state, action)

        q_next = np.max(self.approximator.get(next_state)) if not terminal else 0.

        td_value = q_current + self._alpha * (reward + self.env_info.gamma * q_next - q_current)
        
        self.approximator.set(td_value, state, action)