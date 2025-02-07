import numpy as np
from minirl.rl_utils.parameters import Parameter


def test_parameter():
    parameter = Parameter(1.0, decay=0.95, min_value=0.5)
    for _ in range(50):
        value = parameter()
        print(value)