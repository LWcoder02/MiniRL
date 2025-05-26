import numpy as np

import torch
import torch.nn as nn

from .policy import Policy


class AbstractTorchPolicy(Policy):
    def __init__(self):
        super().__init__()


    def draw_action(self, state):
        with torch.no_grad():
            action = self._draw_action_tensor(state)

        return torch.squeeze(action, dim=0).detach()
    

    def _draw_action_tensor(self, state):
        raise NotImplementedError