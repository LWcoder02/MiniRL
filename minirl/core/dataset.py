from __future__ import annotations
from typing import Tuple

from minirl.core.serialization import Serialization
from minirl.core.datasets.numpy_dataset import NumpyDataset
from .backend import Backend
from minirl.core.environment import EnvironmentInfo
from minirl.core.agent import AgentInfo


class DatasetInfo(Serialization):
    def __init__(self, backend: str, device: str, horizon: float, gamma: float,
                 state_shape: Tuple, action_shape: Tuple):

        self.backend = backend
        self.device = device
        self.gamma = gamma
        self.horizon = horizon
        self.state_shape = state_shape
        self.action_shape = action_shape
        super().__init__()


    @staticmethod
    def create_dataset_info(environment_info: EnvironmentInfo, device: str = None) -> DatasetInfo:
        backend = environment_info.backend
        horizon = environment_info.horizon
        gamma = environment_info.gamma
        state_shape = environment_info.observation_space.shape
        action_shape = environment_info.action_space.shape
        return DatasetInfo(backend=backend, device=device, horizon=horizon,
                           gamma=gamma, state_shape=state_shape, action_shape=action_shape)
    


class Dataset(Serialization):
    def __init__(self, dataset_info: DatasetInfo, num_steps: int = None, num_episodes: int = None):
        
        self._dataset_info = dataset_info
        self._backend: Backend = Backend.get_backend(backend_type=self._dataset_info.backend)


        if num_steps is not None:
            num_samples = num_steps
        else:
            horizon = self._dataset_info.horizon
            num_samples = horizon * num_episodes

        base_shape = (num_samples,)
        state_shape = base_shape + self._dataset_info.state_shape
        action_shape = base_shape # + self._dataset_info.action_shape, has to be checked for some environments
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
    

    @classmethod
    def create_new_empty_dataset(cls, dataset: Dataset = None) -> Dataset:
        new_dataset = cls.__new__(cls)
        if dataset is not None:
            new_dataset._backend = dataset._backend
            new_dataset._dataset_info = dataset._dataset_info
        else:
            new_dataset._dataset_info = None

        new_dataset._dataset = None
        return new_dataset
    

    @classmethod
    def generate(cls, environment_info, num_steps = None, num_episodes = None):
        dataset_info: DatasetInfo = DatasetInfo.create_dataset_info(environment_info=environment_info)
        return cls(dataset_info, num_steps, num_episodes)    


    def get_view(self, index, copy: bool = False):
        dataset = self.create_new_empty_dataset(dataset=self)
        dataset._dataset = self._dataset.get_view(index=index, copy=copy)
        return dataset


    def append(self, sample):
        self._dataset.append(*sample)


    def clear(self):
        self._dataset.clear()


    def parse(self, to):
        return self._convert(self.state, self.action, self.reward, self.next_state, self.done,
                               to=to)


    @property
    def state(self):
        return self._dataset.state
    
    @property
    def action(self):
        return self._dataset.action
    
    @property
    def reward(self):
        return self._dataset.reward
    
    @property
    def next_state(self):
        return self._dataset.next_state
    
    @property
    def done(self):
        return self._dataset.done
    

    def _convert(self, *arrays, to="numpy"):
        if to == 'numpy':
            return self._backend.arrays_to_numpy(*arrays)
        else:
            raise NotImplementedError
