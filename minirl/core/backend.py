import numpy as np
import torch
from collections import deque
from typing import List, Any, Tuple


class Backend():

    @staticmethod
    def get_backend(backend_type):
        if backend_type == 'numpy':
            return NumpyBackend
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
    
    @staticmethod
    def to_numpy(array: list| np.ndarray | torch.Tensor):
        raise NotImplementedError
    

    @staticmethod
    def to_torch(array: list| np.ndarray | torch.Tensor):
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



class TorchBackend(Backend):
    @staticmethod
    def to_numpy(array: torch.Tensor) -> np.ndarray:
        return None if array is None else array.detach().cpu().numpy()
    

    @staticmethod
    def to_torch(array: List[Any] | np.ndarray) -> torch.Tensor:
        return array
    

    @classmethod
    def zeros(cls, *dims, dtype=float, device=None)-> torch.Tensor:
        return cls.zeros(dims, dtype=dtype, device=device)
    

    @classmethod
    def ones(cls, *dims, dtype=float, device=None) -> torch.Tensor:
        return cls.ones(dims, dtype=dtype)
    

    @staticmethod
    def size(arr: torch.Tensor) -> torch.Size:
        return arr.size()