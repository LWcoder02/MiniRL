import numpy as np
from abc import ABC, abstractmethod


class Policy(ABC):
    

    @abstractmethod
    def draw_action(self, state) -> int | np.ndarray:
        raise NotImplementedError()
    

    @abstractmethod
    def set_approximator(self, approximator):
        raise NotImplementedError
    
    @abstractmethod
    def draw_action(self, state):
        raise NotImplementedError