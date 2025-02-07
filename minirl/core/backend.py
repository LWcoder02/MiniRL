import numpy as np
import torch
from collections import deque


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
