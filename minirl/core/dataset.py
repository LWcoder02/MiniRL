from minirl.core.serialization import Serialization
from minirl.core.datasets.numpy_dataset import NumpyDataset


class DatasetInfo(Serialization):
    def __init__(self, environment_info, agent_info):

        self.backend = agent_info.backend
        self.device = agent_info.device
        self.gamma = environment_info.gamma


        super().__init__()


class Dataset(Serialization):
    def __init__(self, environment_info, agent_info):
        
        self._dataset_info = DatasetInfo(environment_info, agent_info, agent_info.backend, agent_info.device)


        if self._dataset_info.backend == 'numpy':
            self._dataset = NumpyDataset()
        elif self._dataset_info.backend == 'torch':
            self._dataset = ...
        else:
            self._dataset = ...

        super().__init__()



    @classmethod
    def generate(cls, environment_info, agent_info, backend, device):
        dataset_info = DatasetInfo(environment_info, agent_info, backend=backend, device=device)

        return cls(dataset_info)
    

    def append(self, sample):
        self._dataset.append(sample)


    def clear(self):
        self._dataset.clear()