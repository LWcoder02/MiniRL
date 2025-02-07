from tqdm import tqdm

class TrainLogic():
    def __init__(self):
        self.move_condition = None
        self.fit_condition = None

        self._steps_counter = 0
        self._episodes_counter = 0

        self._steps_bar: tqdm = None
        self._episode_bar: tqdm = None


    def init_learn(self, num_steps_per_fit, num_episodes_per_fit):
        self.num_steps_per_fit = num_steps_per_fit
        self.num_episodes_per_fit = num_episodes_per_fit

        if self.num_steps_per_fit is not None:
            self.fit_condition = self._fit_steps
        else:
            self.fit_condition = self._fit_episodes


    def init_run(self, num_steps, num_episodes, initial_state=None, verbose=True):
        self._num_steps = num_steps
        self._num_episodes = num_episodes

        if num_steps is not None:
            self.move_condition = self._move_steps_condition

        else:
            self.move_condition = self._move_episodes_condition


    def init_evaluate(self):
        self.fit_condition = lambda: False


    def after_step(self, done):
        self._steps_counter += 1

        # self._steps_bar.update(1)

        if done:
            self._episodes_counter += 1
        #     self._episode_bar.update(1)


    def get_initial_state(self, initial_state):
        if initial_state is None:
            return None


    def _move_steps_condition(self):
        return self._steps_counter < self._num_steps
    

    def _move_episodes_condition(self):
        return self._episodes_counter < self._num_episodes


    def _fit_steps(self):
        return self._steps_counter >= self.num_steps_per_fit
    
    
    def _fit_episodes(self):
        return self._episodes_counter >= self.num_episodes_per_fit