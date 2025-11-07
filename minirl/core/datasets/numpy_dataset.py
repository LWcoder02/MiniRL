from __future__ import annotations
from minirl.core.serialization import Serialization
import numpy as np


from typing import Tuple, Dict


class NumpyDataset(Serialization):
    def __init__(self,
                 base_shape: Tuple[int],
                 field_specs: Dict[str, Tuple[Tuple[int, ...], np.dtype]]):
        self._base_shape = base_shape
        self._field_specs = field_specs
        self._field_names = list(field_specs.keys())
        self._data: Dict[str, np.ndarray] = {}

        for field_name, (shape, dtype) in field_specs.items():
            field_shape = base_shape + shape
            self._data[field_name] = np.empty(shape=field_shape, dtype=dtype)

        self._len = 0


    def __repr__(self):
        data_string = "\n"
        for field_name in self._field_names:
            data_string += f"{field_name}:\n" + str(self._data[field_name]) + "\n"
        return f"NumpyDataset: {data_string}"

    
    def __len__(self):
        return self._len
    

    def __getitem__(self, idx):
        items = []
        for field_name in self._field_names:
            items.append(self._data[field_name][idx])
        return tuple(items)
    
    @classmethod
    def create_new_empty_dataset(cls, dataset=None):
        new_dataset = cls.__new__(cls)

        new_dataset._states = None
        new_dataset._actions = None
        new_dataset._rewards = None
        new_dataset._next_states = None
        new_dataset._dones = None
        new_dataset._len = None

        return new_dataset
    

    def append(self, **kwargs):
        i = self._len

        for name, value in kwargs.items():
            self._data[name][i] = value

        self._len += 1


    def clear(self):
        for field_name, (shape, dtype) in self._field_specs:
            field_shape = self._base_shape + shape
            self._data[field_name] = np.empty(shape=field_shape, dtype=dtype)


        self._len = 0
    

    def get_view(self, index, copy: bool = False) -> NumpyDataset:
        new_dataset = self.create_new_empty_dataset(self)

        if copy:
            new_dataset._states = np.empty_like(self._states)
            new_dataset._actions = np.empty_like(self._actions)
            new_dataset._rewards = np.empty_like(self._rewards)
            new_dataset._next_states = np.empty_like(self._next_states)
            new_dataset._dones = np.empty_like(self._dones)

            new_states = self._states[index, ...]
            new_len = new_states.shape[0]

            new_dataset._states[:new_len] = new_states[index, ...]
            new_dataset._actions[:new_len] = self._actions[index, ...]
            new_dataset._rewards[:new_len] = self._rewards[index, ...]
            new_dataset._next_states[:new_len] = self._next_states[index, ...]
            new_dataset._dones[:new_len] = self._dones[index, ...]
            new_dataset._len = new_len

        else:
            for field_name in self._field_names:
                new_dataset.set_field_value(field_name=field_name,
                                            value=self._data[field_name][index, ...])
                
            new_dataset._len = new_dataset._data[self._field_names[0]].shape[0]

        return new_dataset
    

    def get_field_values(self, field_name):
        return self._data[field_name]
    

    def set_field_value(self, field_name, value):
        self._data[field_name] = value
