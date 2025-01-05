import numpy as np
from abc import ABC, abstractmethod


class AbstractPolicy(ABC):
    

    @abstractmethod
    def draw_action(self, state) -> int | np.ndarray:
        raise NotImplementedError()