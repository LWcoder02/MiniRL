import torch
import numpy as np

from minirl.approximators.approximator import Approximator
from minirl.utils.minibatch_handling import generate_minibatch
import torch.nn as nn


class TorchApproximator(Approximator):
    def __init__(self, model: nn.Module, input_shape: int, output_shape: int,
                 optimizer: torch.optim.Optimizer | dict = None,
                 loss_function: torch.nn.Module | str = None,
                 num_fit_args: int = 1):
        self.model: nn.Module = model(input_shape, output_shape)
        self._num_fit_args = num_fit_args

        self._optimizer: torch.optim.Optimizer = getattr(torch.optim, optimizer['class']) if type(optimizer) is dict else optimizer['class']
        self._loss_function: torch.nn.Module = getattr(torch.nn, loss_function) if type(loss_function) is str else loss_function
        self._optimizer(self.model.parameters(), **optimizer['params'])


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
    

    def set_weights(self, weights: torch.Tensor):
        idx = 0
        for params in self.model.parameters():
            shape = params.data.shape

            c = 1
            for s in shape:
                c *= s

            weight = weights[idx:idx + c].reshape(shape)

            weight_tensor = torch.as_tensor(weight, device=self._device).type(params.data.dtype)

            params.data = weight_tensor
            idx += c


    def get_weights(self):
        weights = list()

        for p in self.model.parameters():
            weight = p.data.detach()
            weights.append(weight.flatten())

        weights = torch.concatenate(weights)
        return weights
