import torch
from minirl.algorithms.dqn.abstract_dqn import AbstractDQN


class DQN(AbstractDQN):
    def _q_next(self, next_state, terminal: torch.Tensor):
        q: torch.Tensor = self.target_approximator.predict(next_state)
        if terminal.any():
            q *= 1 - terminal.reshape(-1, 1)

        return q.max(1)
