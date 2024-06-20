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

        return self._move(move_condition=move_condition,
                  fit_condition=fit_condition,
                  verbose=verbose)
        

    def evaluate(self, move_condition=None, initial_state=None, verbose=True):
        if move_condition is None:
            move_condition = self._logic.move_condition

        fit_condition = self._logic.fit_condition

        return self._move(move_condition, fit_condition,
                          initial_state, verbose=verbose)


    def _move(self, move_condition, fit_condition, init_state=None, verbose=True):
        self._logic.init_run(init_state, verbose)

        rollout = []

        terminal = True
        while move_condition():
            if terminal:
                state = self._reset(init_state)

            sample = self.step(state)

            if fit_condition():
                self._train(rollout)

                rollout.clear()

            terminal = sample[5]
            state = sample[3]

        self.env.stop()

        return rollout


    def _train(self):
        raise NotImplementedError("Train method is not implemented since Agent is an abstract class")


    def draw_action(self, state):
        pass


    def _reset(self):
        pass


    def step(self, state):
        assert self.env is not None, "Cannot move, environment is None"

        action = self.draw_action(state)
        next_state, reward, absorbing, info = self.env.step(action)

        last = False
        return (state, action, reward, next_state, absorbing, last)