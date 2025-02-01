import torch
import numpy as np

from minirl.approximators.approximator import Approximator
from minirl.utils.minibatch_handling import generate_minibatch


class TorchApproximator(Approximator):
    def __init__(self):
        self._optimizer: torch.optim.Optimizer = None
        self._loss_function: torch.nn.Module = None


    def predict(self) -> torch.Tensor:
        pass


    def fit(self, *args, epochs, batch_size, num_targets=1):
        self._batch_size = batch_size
        for epoch in range(epochs):
            batches = generate_minibatch(self._batch_size, *args,
                                              num_targets=num_targets)
            loss = self._perform_training_epoch(batches)


    def _perform_training_epoch(self, batches):
        loss_epoch = list()
        for sample, target in batches:
            loss: torch.Tensor = self._compute_batch_loss(sample, target)
            loss_epoch.append(loss.item())

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()


    def _compute_batch_loss(self, samples, targets):
        pass