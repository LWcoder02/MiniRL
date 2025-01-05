from minirl.core.agent import Agent
from minirl.core.dataset import Dataset
from minirl.approximators.approximator import Approximator


class TD(Agent):

    def __init__(self,
                 env_info,
                 policy,
                 approximator: Approximator,
                 learning_rate: float):

        self._alpha = learning_rate
        policy.set_q(approximator)
        self.approximator = approximator

        super().__init__(env_info, policy)


    def train(self, dataset: Dataset):
        assert len(dataset) == 1

        state, action, reward, next_state, done, _ = dataset[0]
        self._update(state, action, reward, next_state, done)


    def _update(self, state, action, reward, next_state, done):
        pass