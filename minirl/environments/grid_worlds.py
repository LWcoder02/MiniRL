import numpy as np
import gymnasium as gym
from gymnasium import spaces

from minirl.core.environment import Environment, EnvironmentInfo

class GridWorld(Environment):
    def __init__(self, size=(5,5),
                 start_position = np.array([0,0]),
                 target_position = np.array([4,4]),
                 render_mode=None):
        self.size = size
        self.render_mode = render_mode

        self.start_position: np.ndarray[int, int] = np.array([0, 0])
        self._target_position: np.ndarray[int, int] = np.array([self.size[1]-1, self.size[0]-1])

        self.observation_space = spaces.Box(low= np.array([0,0]),
                                            high=np.array(self.size)-1,
                                            shape=(2,), dtype=int)

        self._maze = np.zeros(size)
        self._maze[self._target_position[1], self._target_position[0]] = 2
        self._maze[self.start_position[1], self.start_position[0]] = 1

        self.action_space = spaces.Discrete(4)

        self.state: np.ndarray[int, int] = self.start_position

        self._actions = {
            0: np.array([1,0]), # Right
            1: np.array([-1,0]), # Left
            2: np.array([0,-1]), # Up
            3: np.array([0,1]) # Down
        }

        env_info = EnvironmentInfo(observation_space=self.observation_space,
                                   action_space=self.action_space)
        super().__init__(env_info)
    


    def render(self):
        grid = np.copy(self._maze)
        x, y = self.state[0], self.state[1]
        goal_x, goal_y = self._target_position[0], self._target_position[1]
        grid[goal_x, goal_y] = 2
        grid[y,x] = 1
        print(grid)


    def reset(self, seed = 0, initial_state=None):
        self._maze = np.zeros(self.size)
        self.state = self.start_position
        return self.state, {}


    def step(self, action:int):
        reward = -1
        done = False

        next_state = self.state.copy() + self._actions[action]
        if (0 <= next_state[0] < self.size[0]) and (0 <= next_state[1] < self.size[1]):
            if self._maze[next_state[1], next_state[0]] != -1:
                self.state = next_state
            else:
                reward = -10
        else:
            reward = -10

        done = np.all(self.state == self._target_position)
        reward = 1 if done else reward

        return next_state, reward, done, self._get_info()


    def _get_obs(self):
        return self.state
    

    def _get_info(self):
        return {
            'distance': np.linalg.norm(
                self.state - self._target_position, ord=1
            ),
            'agent': self.state,
            'target': self._target_position
        }