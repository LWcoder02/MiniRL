import sys
sys.path.insert(0, '../')

import numpy as np

from minirl.environments.multi_agent_environments.board_game_environments.tictactoe_env import TicTacToe

def test_environemt():
    env = TicTacToe()
    state = env._state
    print(env)
    sarsa = env.step(4)
    env.step(5)
    env.step(6)
    env.step(8)
    sarsa = env.step(1)
    sarsa = env.step(2)
    print(sarsa)
    print(env)


if __name__ == '__main__':
    test_environemt()