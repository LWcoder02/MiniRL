from collections import defaultdict
import numpy as np
from minirl.approximators.approximator import Approximator


class Table(Approximator):
    def __init__(self, num_actions: int, initial_value: float = 0.):
        self.table = defaultdict(lambda: np.ones(shape=(num_actions,)) * initial_value)
        self.num_actions = num_actions



    def __repr__(self):
        if len(self.table) == 0:
            return "{}"
        dict_string = "{\n"
        for key, value in self.table.items():
            dict_string += f"{key}: {value} \n"
        dict_string += "}"
        return dict_string
    

    def __len__(self):
        return len(self.table)


    def set(self, value, state, action=None):
        state = tuple(state)
        if action is None:
            self.table[state][:] = value
        else:
            self.table[state][action] = value


    def get(self, state, action=None):
        """
        returns value for action and state
        """
        state_key = tuple(state)
        if action == None:
            return self.table[state_key]

        return self.table[state_key][action]
    

    def predict(self, state):
        return self.get(state)
