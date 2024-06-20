from minirl.core.serialization import Serialization
from MiniRL.minirl.core.logic.run_logic import TrainLogic


class Agent(Serialization):
    def __init__(self, policy, environment=None):
        self.policy = policy
        self.env = environment

        self._logic = TrainLogic()


    def learn(self, environment=None, move_condition=None, fit_condition=None, verbose=True):
        if self.env is None and environment is not None:
            self.env = environment
        else:
            raise RuntimeError("Environment is None")

        if move_condition is None:
            move_condition = self._logic.move_condition
        if fit_condition is None:
            fit_condition = self._logic.fit_condition

        self._move(move_condition=move_condition,
                  fit_condition=fit_condition,
                  verbose=verbose)
        

    def _move(self, move_condition, fit_condition, init_state=None, verbose=True):
        self._logic.init_run(init_state, verbose)

        rollout = []

        terminal = True
        while move_condition():
            if terminal:
                self._reset(init_state)

            sample = self.step()

            if fit_condition():
                self._train(rollout)


                rollout.clear()

            terminal = sample[5]

        
        self.stop()
        self.env.stop()

        return rollout


    def evaluate(self):
        pass


    def _train(self):
        pass


    def draw_action(self):
        pass


    def stop(self):
        pass


    def _reset(self):
        pass


    def step(self):
        pass