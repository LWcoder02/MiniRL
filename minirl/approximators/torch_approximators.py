import torch
import numpy as np

from minirl.approximators.approximator import Approximator
from minirl.utils.minibatch_handling import generate_minibatch
import torch.nn as nn


class TorchApproximator(Approximator):
    def __init__(self, model: nn.Module, input_shape: int, output_shape: int, num_fit_args: int = 1):
        self._optimizer: torch.optim.Optimizer = None
        self._loss_function: torch.nn.Module = None
        self.model = model(input_shape, output_shape)
        self._num_fit_args = num_fit_args


    def predict(self, *args) -> torch.Tensor:
        predict_args = [torch.as_tensor(x) for x in args]
        val = self.model(*predict_args)
        return val


    def fit(self, *args, epochs, batch_size, num_targets=1):
        self._batch_size = batch_size
        for epoch in range(epochs):
            batches = generate_minibatch(self._batch_size, *args,
                                              num_targets=num_targets)
            loss_epoch = self._perform_training_epoch(batches)


    def _perform_training_epoch(self, batches):
        loss_epoch = list()
        for batch in batches:
            loss: torch.Tensor = self._compute_batch_loss(batch)
            loss_epoch.append(loss.item())

            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()

        return np.mean(loss_epoch)


    def _compute_batch_loss(self, batch):
        training_args = [torch.as_tensor(x) for x in batch]

        x = training_args[:-self._num_fit_args]
        y = [y_i.clone().detach() for y_i in training_args[-self._num_fit_args:]]

        yhat = self.model(*x)
        loss = self._loss_function(yhat, *y)

        return loss
