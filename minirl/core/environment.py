import numpy as np
from typing import Tuple, Any, Dict

from minirl.core.serialization import Serialization


class EnvironmentInfo(Serialization):
    def __init__(self, action_space, observation_space, gamma: float = 0.9, horizon: int = None,
                 backend: str = 'numpy'):
        self.gamma = gamma
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_actions = self.action_space.n
        self.horizon = horizon
        self.backend = backend


class Environment(object):
    def __init__(self, environment_info: EnvironmentInfo):
        self._environment_info = environment_info


    def get_environment_info(self) -> EnvironmentInfo:
        return self._environment_info


    def make(self, env_name):
        pass


    def seed(self, seed):
        pass


    def reset(self, seed: int = 0, initial_state=None):
        pass


    def step(self, action: int) -> Tuple[Any, float, bool, Dict[Any, Any]]:
        pass


    def render(self):
        pass


    def get_opponent(self):
        pass


    def get_opponent_value(self, value):
        pass