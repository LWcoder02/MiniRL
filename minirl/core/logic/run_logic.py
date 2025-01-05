

class TrainLogic():
    def __init__(self):
        self.move_condition = None
        self.fit_condition = None

        self._steps_counter = 0
        self._episodes_counter = 0


    def init_learn(self, num_steps_per_fit, num_episodes_per_fit):
        self.num_steps_per_fit = num_steps_per_fit
        self.num_episodes_per_fit = num_episodes_per_fit

        if self.num_steps_per_fit is not None:
            self.fit_condition = self._fit_steps
        else:
            self.fit_condition = self._fit_episodes


    def init_run(self, initial_state=None, verbose=True):
        pass


    def init_evaluate(self):
        self.fit_condition = lambda: False


    def _fit_steps(self):
        return self._steps_counter >= self.num_steps_per_fit
    
    
    def _fit_episodes(self):
        return self._episodes_counter >= self.num_episodes_per_fit