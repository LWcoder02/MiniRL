from minirl.core.agent import Agent
from minirl.core.dataset import Dataset
from minirl.approximators.table import Table
from minirl.policy.policy import Policy
from minirl.core.environment import EnvironmentInfo


class TD(Agent):

    def __init__(self,
                 environment_info: EnvironmentInfo,
                 policy: Policy,
                 approximator: Table,
                 learning_rate: float,
                 agent_id: int | str = "agent_0"):


        self._alpha: float = learning_rate
        policy.set_approximator(approximator)
        self.approximator: Table = approximator

        super().__init__(environment_info, policy, agent_id)


    def train(self, dataset: Dataset):
        assert len(dataset) == 1

        state, action, reward, next_state, done = dataset[0]
        self._update(state, action, reward, next_state, done)


    def _update(self, state, action, reward, next_state, done):
        pass