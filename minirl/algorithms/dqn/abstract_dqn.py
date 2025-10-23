from copy import deepcopy

import numpy as np

from minirl.core.agent import Agent
from minirl.approximators.torch_approximators import TorchApproximator
from minirl.rl_utils.replay_memory import ReplayBuffer

from minirl.policy.policy import Policy
from typing import Callable


class AbstractDQN(Agent):
    def __init__(self, environment, policy: Policy, approximator_params: dict, approximator: TorchApproximator = TorchApproximator, batch_size=32, target_update_frequency=50,
                 initial_replay_size=500, max_replay_size=5000, clip_reward=False):
        
        self._replay_buffer = ReplayBuffer()
        self._fit: Callable = self._train_standard


        approximator_params: dict = deepcopy(approximator_params)
        approximator_params_target: dict = deepcopy(approximator_params)
        self._initialize_approximator(approximator, approximator_params, approximator_params_target)

        self._num_updates: int = 0
        self._clip_reward: bool = clip_reward
        self._update_frequency: int = target_update_frequency
        self._batch_size: int = batch_size

        policy.set_approximator(self.approximator)
        super().__init__(environment=environment, policy=policy)


    def _train(self, dataset):
        self._fit(dataset)

        self._num_updates += 1
        if self._num_updates % self._update_frequency == 0:
            self._update_target()


    def _train_standard(self, dataset):
        self._replay_buffer.add(dataset)
        if self._replay_buffer.init:
            state, action, reward, next_state, terminal, info = self._replay_buffer.sample(self._batch_size)

            # clip reward here
            if self._clip_reward:
                reward = np.clip(reward, -1, 1)

            q_next = self._q_next(next_state, terminal)
            q = reward + self._env_info.gamma * q_next


            pred = self.approximator.predict(state).gather(dim=1, index=action)
            self.approximator.fit(pred, q)


    def _q_next(self, next_state, terminal):
        raise NotImplementedError()
    

    def _initialize_approximator(self, approximator, approximator_params, approx_params_target):
        self.approximator: TorchApproximator = approximator(approximator_params)
        self.target_approximator: TorchApproximator = approximator(approx_params_target)
        self._update_target()


    def _update_target(self):
        self.target_approximator.set_weights(self.approximator.get_weights())