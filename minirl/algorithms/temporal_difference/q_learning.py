import numpy as np

from minirl.algorithms.temporal_difference.temporal_difference import TD
from minirl.approximators.table import Table
from minirl.core.environment import EnvironmentInfo
from minirl.policy.policy import Policy


class QLearning(TD):
    def __init__(self, environment_info: EnvironmentInfo, policy: Policy, learning_rate: float,
                 agent_id: int | str = "agent_0"):

        env_info: EnvironmentInfo = environment_info

        approximator = Table(env_info.num_actions)

        super().__init__(env_info, policy, approximator, learning_rate, agent_id)


    def _update_experimental(self, state, action, reward, next_state, done):
        q_current = self.approximator.get(state, action)

        q_next = np.max(self.approximator.get(next_state)) if not done else 0.

        update_value = q_current + self._alpha(state, action) * (
            reward + self._env_info.gamma * q_next - q_current
        )

        self.approximator.set(update_value, state, action)


    def _update(self, state, action, reward, next_state, terminal):
        q_current = self.approximator.get(state, action)

        q_next = np.max(self.approximator.get(next_state)) if not terminal else 0.

        td_value = q_current + self._alpha * (reward + self._env_info.gamma * q_next - q_current)
        
        self.approximator.set(td_value, state, action)