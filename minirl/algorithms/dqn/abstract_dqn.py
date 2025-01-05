from copy import deepcopy

import numpy as np
import torch

from minirl.core.agent import Agent
from minirl.approximators.torch_approximators import TorchApproximator


class AbstractDQN(Agent):
    def __init__(self):
        
        self._replay_buffer = []
        self._fit = self._train_standard

        self.approximator = TorchApproximator
        self.target_approximator = TorchApproximator


    def _train(self, dataset):
        self._fit(dataset)


    def _train_standard(self, dataset):
        self._replay_buffer.add(dataset)
        if self._replay_buffer.init:
            state, action, reward, next_state, terminal, info = self._replay_buffer.sample()

            # clip reward here

            q_next = self._q_next(next_state, terminal)
            q = reward + self.env_info.gamma * q_next


            pred = self.approximator.predict(state).gather(dim=1, index=action)
            self.approximator.train(pred, q)


    def _q_next(self, next_state, terminal):
        raise NotImplementedError()