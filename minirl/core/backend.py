import numpy as np
import torch
from collections import deque


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