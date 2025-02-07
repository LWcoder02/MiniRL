import numpy as np



class Parameter():
    def __init__(self, value, min_value=None, max_value = None, decay = None, num_steps = -1, parameter_type = 'constant'):
        self._initialize_values(value=value,
                             min_value=min_value,
                             max_value=max_value,
                             decay = decay,
                             num_steps = num_steps,
                             parameter_type= parameter_type)


    def _initialize_values(self, value, min_value, max_value, decay, num_steps, parameter_type):
        self._initial_value = value
        self._value = value
        self._min_value = min_value
        self._max_value = max_value
        self._decay = decay
        self._steps = num_steps
        self._type = parameter_type
        self._num_updates = 0


        min_max_value = min_value
        if min_value is not None and max_value is not None:
            raise ValueError()
        elif max_value is not None:
            min_max_value = max_value

        self._coefficient = (min_max_value - value) / num_steps if min_max_value is not None else None


        if self._type == 'constant' and self._decay is None:
            self._compute = self._compute_constant
        elif self._type == 'linear' or self._decay is not None:
            self._compute = self._compute_linear
        


    def _compute_constant(self):
        return self._initial_value


    def _compute_linear(self):
        if self._decay is not None:
            self._value = self._decay * self._value
            return self._value
        if self._coefficient is None:
            raise ValueError("Linear parameter computation cannot be performed, either max_value or min_value is None")
        return self._coefficient * self._num_updates + self._initial_value


    def _compute_exponential(self):
        raise NotImplementedError()



    def __call__(self):
        self._update()

        return self.get_value()
    

    def _update(self):
        self._num_updates += 1


    def get_value(self):
        new_value = self._compute()

        if self._min_value is None and self._max_value is None:
            return new_value
        
        return np.clip(new_value, self._min_value, self._max_value)


