from minirl.core.agent import Agent
from minirl.core.dataset import Dataset
from minirl.approximators.table import Table


class TD(Agent):

    def __init__(self,
                 environment,
                 policy,
                 approximator: Table,
                 learning_rate: float):


        self._alpha: float = learning_rate
        policy.set_approximator(approximator)
        self.approximator: Table = approximator

        super().__init__(environment, policy)


    def train(self, dataset: Dataset):
        assert len(dataset) == 1

        state, action, reward, next_state, done = dataset[0]
        self._update(state, action, reward, next_state, done)


    def _update(self, state, action, reward, next_state, done):
        pass