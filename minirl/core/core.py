from minirl.core.logic.run_logic import TrainLogic
from minirl.core.agent import Agent
from minirl.core.dataset import Dataset


class Core():
    def __init__(self, agent: Agent, environment):
        self.agent = agent
        self.environment = environment

        self._logic = TrainLogic()

        self._state = None




    def learn(self, num_steps=None, num_episodes=None,
              num_steps_per_fit=None, num_episodes_per_fit=None, quiet=False):
        self._logic.init_learn(num_steps_per_fit=num_steps_per_fit, num_episodes_per_fit=num_episodes_per_fit)

        dataset = Dataset()

        self._run_impl(dataset)


    def evaluate(self, num_steps=None, num_episodes=None, quiet=False):
        self._logic.init_evaluate()

        dataset = Dataset()

        return self._run_impl(dataset)


    def _run_impl(self, dataset):
        self._logic.init_run()

        done = True
        while self._logic.move_condition():
            if done:
                self._reset()


            sample = self._step()

            dataset.append(sample)


            if self._logic.fit_condition():
                self.agent.train(dataset)

                dataset.clear()

            
            done = sample[4]


        return dataset
    

    def _step(self):
        action = self.agent.draw_action(self._state)
        next_state, reward, done, info = self.environment.step(action)

        state = self._state
        self._state = next_state

        return (state, action, reward, next_state, done), info


    def _reset(self):
        pass