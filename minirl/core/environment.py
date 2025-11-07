import numpy as np
import gymnasium as gym
from typing import Tuple, Any, Dict, TypeVar

from minirl.core.serialization import Serialization
from abc import abstractmethod, ABC



AgentID = TypeVar("AgentID")
ActionType = TypeVar("ActionType")
ObsType = TypeVar("ObsType")
ObsDict = Dict[AgentID, ObsType]
ActionDict = Dict[AgentID, ActionType]
RewardDict = Dict[AgentID, float]
TerminatedDict = Dict[AgentID, bool]
InfoDict = Dict[AgentID, Dict[str, Any]]


class EnvironmentInfo(Serialization):
    def __init__(self, action_space, observation_space, gamma: float = 0.9, horizon: int = None,
                 backend: str = 'numpy'):
        self.gamma = gamma
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_actions = self.action_space.n
        self.horizon = horizon
        self.backend = backend


class Environment(ABC):
    def __init__(self, environment_info: EnvironmentInfo):
        self._environment_info = environment_info


    @property
    def environment_info(self) -> EnvironmentInfo:
        return self._environment_info


    def make(self, env_name):
        pass


    def seed(self, seed):
        pass


    @property
    @abstractmethod
    def action_space(self) -> gym.spaces.Dict:
        pass


    @property
    @abstractmethod
    def observation_space(self) -> gym.spaces.Dict:
        pass


    @ abstractmethod
    def reset(
        self,
        seed: int = 0,
        initial_state=None,
        options: Dict[str, Any] | None = None
    ) -> Tuple[ObsDict, InfoDict]:
        pass


    @abstractmethod
    def step(
            self,
            action: ActionDict | ActionType
    ) -> Tuple[ObsDict, RewardDict, TerminatedDict, InfoDict]:
        raise NotImplementedError()


    @abstractmethod
    def render(self) -> None:
        pass


    @abstractmethod
    def close(self) -> None:
        pass
    

    # def get_opponent(self):
    #     pass


    # def get_opponent_value(self, value):
    #     pass

