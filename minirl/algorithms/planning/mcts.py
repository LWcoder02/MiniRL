import numpy as np
import math
import random

from minirl.core.agent import Agent


class TreeNode(object):
    def __init__(self, state, available_actions,
                 parent = None, action_taken = None, player = None, terminal = False):
        self._state = state
        self._parent = parent
        self._available_actions = available_actions
        self._action = action_taken
        self._player = player
        self._terminal = terminal

        self._children = []
        self._visit_count = 0
        self._total_reward = 0.0


    def select(self):
        ...


    def expand(self):
        ...


    def simulate(self):
        ...


    def backpropagate(self, value):
        ...



class MonteCarloTreeSearch(Agent):
    def __init__(self, environment, node_type = TreeNode, backend = 'numpy'):
        policy = None
        self._node_type = node_type
        super().__init__(environment, policy, backend)



    def _search(self, state, num_simulations = 1000) -> int:
        root = self._node_type(state=state.copy(), parent=None)

        for _ in range(num_simulations):
            selected_node: TreeNode = root.select()

            expanded_node = selected_node
            if not selected_node._terminal:
                expanded_node = selected_node.expand()

            value = expanded_node.simulate()

            expanded_node.backpropagate(value=value)


        best_action = self._get_best_action(root)
        return best_action
    

    def _get_best_action(self, node):
        ...


    def learn(self, num_steps=None, num_episodes=None,
              num_steps_per_fit=None, num_episodes_per_fit=None,
              quiet=False):
        raise NotImplementedError("MonteCarloTreeSearch does not support learning to move in an environment")
    

    def evaluate(self, num_steps=None, num_episodes=None, quiet=False):
        raise NotImplementedError("Evaluate is currently not implemented")
    

    def draw_action(self, state):
        return self._search(state=state)