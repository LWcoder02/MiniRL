import numpy as np
from gymnasium import spaces
from typing import Tuple, Any, Dict, List
from copy import deepcopy

from minirl.core.environment import EnvironmentInfo
from .board_games import AbstractBoardGame


class GameLogic(object):
    def __init__(self):
        self.size = 3


    def _check_win(self, state, player):
        for i in range(self.size):
            if np.all(state[i, :] == player) or np.all(state[:, i] == player):
                return True, 1
            
        if np.all(np.diag(state) == player) or np.all(np.diag(np.fliplr(state)) == player):
            return True, 1
        return False, 0


    def _check_draw(self, state):
        return np.all(state != 0)
    


class _GameState(object):
    def __init__(self, board_shape: Tuple[int, int] = (3,3),
                 initial_state: np.ndarray[np.int8, np.int8] = None):
        self._board_shape = board_shape
        self._board = np.zeros(self._board_shape, dtype=np.int8) if initial_state is None else initial_state
        
        self._game_logic = GameLogic()


    def __repr__(self):
        return f"Board:\n{self._board}"


    def apply_action(self, action, player: int = None, copy: bool = False):
        if copy:
            state_copy = deepcopy(self)
            return state_copy.apply_action(action=action, player=player, copy=False)
        
        pos = divmod(action, 3)

        self._board[pos] = player
        done, reward = self._game_logic._check_win(self._board, player)
        reward = reward if player == 1 else -reward

        player = self._switch_player(player)
        info_dict = {
            "player_turn": player
        }
        return self, reward, done, info_dict


    def get_observation(self):
        return self._board.flatten()
    

    def _switch_player(self, player):
        return -player


    def change_view(self):
        self._board *= -1


    def get_legal_actions(self, board_state: np.ndarray) -> List[int]:
        return list(np.where(self._board.flatten() == 0)[0])



class TicTacToe(AbstractBoardGame):
    def __init__(self):

        self._board_size = 3
        self._state = _GameState()
        self._current_player = 1

        self._observation_space = spaces.Box(low=-1, high=1,
                                             shape=(self._board_size * self._board_size,),
                                             dtype=np.int8)
        self._action_space = spaces.Discrete(9)

        env_info = EnvironmentInfo(action_space=self._action_space,
                                   observation_space=self._observation_space)
        

        super().__init__(environment_info=env_info)


    def __repr__(self):
        return f"Game State:\n{self._state.get_observation().reshape((3,3))}\nCurrent Player: {self._current_player}"


    def reset(self, seed: int = 0, initial_state = None):
        self._state = _GameState(initial_state=initial_state)
        return self._state.get_observation(), {}

    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, int | np.ndarray]]:
        next_state, reward, terminated, info = self._state.apply_action(action, self._current_player)
        self._current_player = info['player_turn']
        return next_state.get_observation(), reward, terminated, info

    
    def get_legal_actions(self) -> List[int]:
        return self._state.get_legal_actions()
    

    def change_view(self):
        self._state.change_view()
        self._current_player *= -1


    def get_state(self):
        return deepcopy(self._state)