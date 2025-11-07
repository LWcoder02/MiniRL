from __future__ import annotations
from copy import deepcopy
from abc import abstractmethod


class AbstractState(object):
    def __init__(self):
        pass


    def clone(self) -> AbstractState:
        return deepcopy(self)
    

    @abstractmethod
    def get_actions(self):
        pass


    @abstractmethod
    def apply_action(self, action):
        pass


    @abstractmethod
    def reset(self):
        pass