import numpy as np
import torch
from collections import deque


<<<<<<< HEAD
class Backend():
    
    @staticmethod
    def to_numpy():
        raise NotImplementedError
    

    @staticmethod
    def to_torch():
        raise NotImplementedError
    

    @classmethod
    def zeros(cls, *dims, dtype, device=None):
        raise NotImplementedError
    
    @classmethod
    def ones(cls, *dims, dtype, device=None):
        raise NotImplementedError
    

    @staticmethod
    def size(arr):
        raise NotImplementedError
    

class NumpyBackend(Backend):

    @staticmethod
    def to_numpy(array):
        return array
    
    @staticmethod
    def to_torch(array):
        return None if array is None else torch.from_numpy(array)
    

    @classmethod
    def zeros(cls, *dims, dtype=float, device=None):
        return cls.zeros(dims, dtype=dtype)
    
    @classmethod
    def ones(cls, *dims, dtype=float, device=None):
        return cls.ones(dims, dtype=dtype)
    

    @staticmethod
    def size(arr):
        return np.size(arr)
=======
class DatasetBackend(object):

    @staticmethod
    def to_numpy(dataset):
        raise NotImplementedError()
    

    @staticmethod
    def to_torch(dataset):
        raise NotImplementedError()



class NumpyDataset(DatasetBackend):
    
    
    @staticmethod
    def to_numpy(dataset):
        return dataset
    

    def to_torch(dataset):
        return torch.from_numpy(dataset)


class ListDataset(DatasetBackend):
    ...


class TorchDataset(DatasetBackend):
    ...
>>>>>>> 8c1f63ed8883cd2453d62562098f31a259a2d8b9
