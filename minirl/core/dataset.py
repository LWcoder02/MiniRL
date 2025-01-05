from minirl.core.serialization import Serialization
from minirl.core.backend import DatasetBackend


class DatasetInfo(Serialization):
    def __init__(self, environment_info, agent_info, backend, device):

        self.backend = backend
        self.device = device
        self.gamma = environment_info.gamma


        super().__init__()


class Dataset(Serialization):
    def __init__(self, dataset_info):
        

        if dataset_info.backend == 'numpy':
            self._dataset = ...
        elif dataset_info.backend == 'torch':
            self._dataset = ...
        else:
            self._dataset = ...

        self._dataset_info = dataset_info

        super().__init__()



    @classmethod
    def generate(cls, environment_info, agent_info, backend, device):
        dataset_info = DatasetInfo(environment_info, agent_info, backend=backend, device=device)

        return cls(dataset_info)