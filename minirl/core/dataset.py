from minirl.core.serialization import Serialization
from minirl.core.datasets.numpy_dataset import NumpyDataset
from .backend import Backend


class DatasetInfo(Serialization):
    def __init__(self, environment_info, agent_info):

        self.backend = agent_info.backend
        self.device = agent_info.device
        self.gamma = environment_info.gamma
        self.state_shape = environment_info.observation_space.shape
        self.action_shape = environment_info.action_space.n


        super().__init__()


class Dataset(Serialization):
    def __init__(self, environment_info, agent_info, num_steps=None, num_episodes=None):
        
        self._dataset_info = DatasetInfo(environment_info, agent_info)
        self._backend: Backend = Backend.get_backend(backend_type=self._dataset_info.backend)


        if num_steps is not None:
            num_samples = num_steps
        else:
            horizon = environment_info.horizon
            num_samples = horizon * num_episodes

        base_shape = (num_samples,)
        state_shape = base_shape + self._dataset_info.state_shape
        action_shape = base_shape + self._dataset_info.action_shape
        reward_shape = base_shape


        if self._dataset_info.backend == 'numpy':
            self._dataset = NumpyDataset(state_shape=state_shape, action_shape=action_shape, reward_shape=reward_shape,
                                         flag_shape=base_shape)
        else:
            raise ValueError("Only numpy is as dataset type supported")
        # elif self._dataset_info.backend == 'torch':
        #     self._dataset = ...
        # else:
        #     self._dataset = ...

        super().__init__()


    def __repr__(self):
        return self._dataset.__repr__()
    

    def __len__(self):
        return len(self._dataset)
    

    def __getitem__(self, idx):
        return self._dataset[idx]
    

    def get_view(self, index):
        raise NotImplementedError("Refactoring, get_view method is currently not implemented")


    @classmethod
    def generate(cls, environment_info, agent_info, num_steps = None, num_episodes = None):
        return cls(environment_info, agent_info, num_steps, num_episodes)    


    def append(self, sample):
        self._dataset.append(*sample)


    def clear(self):
        self._dataset.clear()


    def parse(self):
        self._dataset._convert()
