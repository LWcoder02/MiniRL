from __future__ import annotations
import numpy as np
from gymnasium import spaces
from minirl.rl_utils.states import AbstractState

from typing import Tuple, Dict, Any

from minirl.core.environment import Environment, EnvironmentInfo
from minirl.core.environment import ActionType, ObsDict, RewardDict, TerminatedDict, InfoDict


class GridState(AbstractState):
    def __init__(self,
                 agent_position: np.ndarray[int, int],
                 target_position: np.ndarray[int, int],
                 size: np.ndarray[int, int],
                 wall_positions: np.ndarray = None):
        self._agent_position = agent_position
        self._target_position = target_position
        self._size = size
        self._walls = wall_positions
        self._reward: int = -1
        self._terminal: bool = False

    def __getitem__(self, idx):
        return self._agent_position[idx]
    

    def apply_action(self, action: np.ndarray) -> None:
        reward: int = -1
        agent_next_position = self._agent_position.copy() + action
        if (0 <= agent_next_position[0] < self._size[0]) and (0 <= agent_next_position[1] < self._size[1]):
            if self._check_walls(position=agent_next_position):
                self._agent_position = agent_next_position
            else:
                reward = -10
        else:
            reward = -10

        self._terminal = np.all(self._agent_position == self._target_position)
        self._reward = 1 if self._terminal else reward

    
    def get_observation(self) -> np.ndarray:
        return self._agent_position
    

    def get_reward(self) -> int:
        return self._reward
    

    def get_terminal(self) -> bool:
        return self._terminal


    def render(self) -> None:
        pass
    

    def _check_walls(self, position):
        if self._walls is None:
            return True
        return False


    def get_info(self) -> Dict[str, Any]:
        return {
            'distance': np.linalg.norm(self._agent_position - self._target_position, ord=1),
            'agent_position': self._agent_position,
            'target': self._target_position
        }


    def get_actions(self):
        return []


class GridWorld(Environment):
    def __init__(self, size=(5,5),
                 start_position = np.array([0,0]),
                 target_position = np.array([4,4]),
                 render_mode=None):
        self.size = size
        self.render_mode = render_mode

        self._state = GridState(agent_position=start_position, target_position=target_position)

        self._observation_space = spaces.Box(low= np.array([0,0]),
                                            high=np.array(self.size)-1,
                                            shape=(2,), dtype=int)

        # self._agent_id = "agent"
        self._action_space = spaces.Discrete(4)

        self._actions = {
            0: np.array([1,0]), # Right
            1: np.array([-1,0]), # Left
            2: np.array([0,-1]), # Up
            3: np.array([0,1]) # Down
        }
        

        env_info = EnvironmentInfo(action_space=self._action_space, observation_space=self._observation_space,
                                   gamma=0.9, horizon=100, backend='numpy')
        super().__init__(env_info)
    

    def render(self):
        pass


    def reset(self,
              seed: int = 0,
              initial_state=None,
              options: Dict[str, Any] | None = None):
        return self._state.reset()


    def step(self, action: ActionType) -> Tuple[np.ndarray, int, bool, InfoDict]:
        direction = self._actions[action]
        self._state.apply_action(action=direction)

        obs = self._state.get_observation()
        reward = self._state.get_reward()
        done = self._state.get_terminal()
        info = self._state.get_info()

        return obs, reward, done, info

    