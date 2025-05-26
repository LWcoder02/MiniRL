import numpy as np
from gymnasium import spaces

from minirl.core.environment import Environment, EnvironmentInfo



class GridState(object):
    def __init__(self, position: np.ndarray[int, int], target_position: np.ndarray[int, int]):
        self._position = position
        self._target_position = target_position


    def __getitem__(self, idx):
        return self._position[idx]


    def apply_action(self, action: np.ndarray[int, int]):
        self._position = self._position + action
        done = np.all(self._position == self._target_position)

        return self, done, self.get_info()


    def get_observation(self):
        return self._position
    

    def get_info(self):
        return {
            'distance': np.linalg.norm(
                self._position - self._target_position, ord=1
            ),
            'agent': self._position,
            'target': self._target_position
        }
    




class GridWorld(Environment):
    def __init__(self, size=(5,5),
                 start_position = np.array([0,0]),
                 render_mode=None):
        self.size = size
        self.render_mode = render_mode

        self._start_position: np.ndarray[int, int] = start_position
        self._target_position: np.ndarray[int, int] = np.array([self.size[1]-1, self.size[0]-1])

        self.observation_space = spaces.Box(low= np.array([0,0]),
                                            high=np.array(self.size)-1,
                                            shape=(2,), dtype=int)

        self._maze = np.zeros(size)
        self._maze[self._target_position[1], self._target_position[0]] = 2
        self._maze[self._start_position[1], self._start_position[0]] = 1

        self.action_space = spaces.Discrete(4)

        self.state: GridState = GridState(position=self._start_position, target_position=self._target_position)

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
        self.state = GridState(position=self._start_position, target_position=self._target_position)
        return self.state.get_observation(), {}


    def step(self, action:int):
        reward = -1

        next_state, done, info = self.state.apply_action(self._actions[action])
        if (0 <= next_state[0] < self.size[0]) and (0 <= next_state[1] < self.size[1]):
            if self._maze[next_state[1], next_state[0]] != -1:
                self.state = next_state
            else:
                reward = -10
        else:
            reward = -10

        reward = 1 if done else reward

        return next_state.get_observation(), reward, done, info
