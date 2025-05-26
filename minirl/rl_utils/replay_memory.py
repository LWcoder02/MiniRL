from minirl.core.dataset import Dataset


class ReplayBuffer():
    def __init__(self, environment_info, agent_info, max_size: int = 5000):
        self._init: bool = False
        self._max_size: int = max_size
        self._idx: int = 0
        self._environment_info = environment_info
        self._agent_info = agent_info

        self.reset()


    def sample(self, batch_size: int):
        indices = self._dataset._backend.randint(0, len(self._dataset), (batch_size,))

        batch = self._dataset[indices]
        return batch.parse()


    def add(self, dataset):
        ...



    def reset(self):
        self._idx = 0
        self._full = False
        self._dataset = Dataset(environment_info=self._environment_info,
                                agent_info=self._agent_info,
                                num_steps=self._max_size)