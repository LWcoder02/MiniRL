import numpy as np
import gymnasium as gym
from gymnasium import spaces

from minirl.core.environment import Environment

class GridWorld(Environment):
    def __init__(self, size=5, target_position=None, render_mode=None):
        self.size = size
        self.render_mode = render_mode

        self._target_position = np.array([size-1, size-1]) if not target_position else target_position

        self.observation_space = spaces.Dict(
            {
                'agent': spaces.Box(low=0, high=size-1, shape=(2,), dtype=int),
                'target': spaces.Box(low=0, high=size-1, shape=(2,), dtype=int)
            }
        )

        self._state = np.zeros((size,size))

        self.action_space = spaces.Discrete(4)

        self._actions = {
            0: np.array([1,0]), # Right
            1: np.array([-1,0]), # Left
            2: np.array([0,1]), # Up
            3: np.array([0,-1]) # Down
        }


    def reset(self, seed: int = 0, initial_state=None):
        # gym.Env.reset(seed=seed if not seed else 42)
        self._agent_location = np.array([0,0]) if not initial_state else initial_state

        obs = self._get_obs()
        info = self._get_info()

        self._state[self._agent_location] = 1
        self._state[self._target_position] = 2

        return obs, info


    def step(self, action):
        direction = self._actions[action]
        self._state[self._agent_location] = 0

        self._agent_location = np.clip(
            self._agent_location + direction, self.size - 1
        )

        self._state[self._agent_location] = 1

        terminated = np.array_equal(self._agent_location, self._target_position)
        reward = 1 if terminated else 0.1
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, info


    def render(self):
        print(self._state)


    def _get_obs(self):
        return {
            'agent': self._agent_location,
            'target': self._target_position
        }    
    

    def _get_info(self):
        return {
            'distance': np.linalg.norm(
                self._agent_location - self._target_position, ord=1
            )
        }