from minirl.core.serialization import Serialization
import numpy as np



class NumpyDataset(Serialization):
    def __init__(self, state_shape, action_shape, reward_shape, flag_shape):

        self._states = np.empty(state_shape)
        self._actions = np.empty(action_shape, dtype=int)
        self._rewards = np.empty(reward_shape)
        self._next_states = np.empty(state_shape)
        # self._terminated = np.empty(flag_shape)
        self._dones = np.empty(flag_shape, dtype=bool)
        self._len = 0


    def __repr__(self):
        return f"States: Shape{self._states.shape}\n{self._states} \
            \nActions: Shape{self._actions.shape} \n{self._actions}"

    
    def __len__(self):
        return self._len
    

    def __getitem__(self, idx):
        return self._states[idx], self._actions[idx], self._rewards[idx], self._next_states[idx], \
            self._dones[idx]
    

    def append(self, state, action, reward, next_state, done):
        i = self._len

        self._states[i] = state
        self._actions[i] = action
        self._rewards[i] = reward
        self._next_states[i] = next_state
        self._dones[i] = done
        # self._terminated[i] = terminated

        self._len += 1


    def clear(self):
        self._states = np.empty_like(self._states)
        self._actions = np.empty_like(self._actions)
        self._rewards = np.empty_like(self._rewards)
        self._next_states = np.empty_like(self._next_states)
        self._dones = np.empty_like(self._dones)
        self._len = 0
