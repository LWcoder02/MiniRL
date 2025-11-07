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
    def get_backend_name():
        raise NotImplementedError()
    
    @staticmethod
    def to_numpy(array: list| np.ndarray | torch.Tensor):
        raise NotImplementedError
    

    @staticmethod
    def to_torch(array: list| np.ndarray | torch.Tensor):
        raise NotImplementedError
    
    
    @classmethod
    def convert_to_backend(cls, array):
        raise NotImplementedError


    @classmethod
    def arrays_to_numpy(cls, *arrays):
        return tuple(cls.to_numpy(array) for array in arrays)
    
    @classmethod
    def arrays_to_torch(cls, *arrays):
        return tuple(cls.to_torch(array) for array in arrays)
    

    @staticmethod
    def zeros(*dims, dtype, device=None):
        raise NotImplementedError
    
    @staticmethod
    def ones(*dims, dtype, device=None):
        raise NotImplementedError
    

    @staticmethod
    def size(arr):
        raise NotImplementedError
    
    @staticmethod
    def randint(low, high, size):
        raise NotImplementedError
    

    @staticmethod
    def empty(shape, dtype):
        raise NotImplementedError
    

    @staticmethod
    def empty_like(array):
        raise NotImplementedError
    

class NumpyBackend(Backend):

    @staticmethod
    def get_backend_name():
        return 'numpy'

    @staticmethod
    def to_numpy(array):
        return array
    
    @staticmethod
    def to_torch(array):
        return None if array is None else torch.from_numpy(array)
    

    @classmethod
    def convert_to_backend(cls, array):
        return cls.to_numpy(array)


    @staticmethod
    def zeros(*dims, dtype=float, device=None):
        return np.zeros(dims, dtype=dtype)
    
    @staticmethod
    def ones(*dims, dtype=float, device=None):
        return np.ones(dims, dtype=dtype)
    

    @staticmethod
    def size(arr):
        return np.size(arr)
    

    @staticmethod
    def shape(arr: np.ndarray):
        return arr.shape
    

    @staticmethod
    def randint(low, high, size):
        return np.random.randint(low, high, size)
    

    @staticmethod
    def empty(shape, dtype):
        return np.empty(shape=shape, dtype=dtype)
    

    @staticmethod
    def empty_like(array):
        return np.empty_like(array)



class TorchBackend(Backend):

    @staticmethod
    def get_backend_name():
        return 'torch'

    @staticmethod
    def to_numpy(array: torch.Tensor) -> np.ndarray:
        return None if array is None else array.detach().cpu().numpy()
    

    @staticmethod
    def to_torch(array: List[Any] | np.ndarray) -> torch.Tensor:
        return array
    

    @classmethod
    def convert_to_backend(cls, array):
        return cls.to_torch(array)


    @staticmethod
    def zeros(*dims, dtype=float, device=None)-> torch.Tensor:
        return torch.zeros(dims, dtype=dtype, device=device)
    

    @staticmethod
    def ones(*dims, dtype=float, device=None) -> torch.Tensor:
        return torch.ones(dims, dtype=dtype)
    

    @staticmethod
    def size(arr: torch.Tensor) -> torch.Size:
        return arr.size()
    

    @staticmethod
    def shape(arr: torch.Tensor) -> Tuple:
        return arr.size()
    

    @staticmethod
    def empty(shape, dtype):
        return torch.empty(shape=shape, dtype=dtype)
    

    @staticmethod
    def empty_like(array):
        return torch.empty_like(array)
    

    @staticmethod
    def randint(low, high, size):
        return torch.randint(low, high, size)