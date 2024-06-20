from MiniRL.minirl.core.logic.run_logic import TrainLogic
from minirl.core.agent import Agent


class Core():
    def __init__(self, agent: Agent, environment):
        self.agent = agent
        self.environment = environment

        self._logic = TrainLogic()


    def learn(self, num_steps=None, num_episodes=None, num_steps_per_fit=None, num_episodes_per_fit=None,
              verbose=True):
        self._logic.init_learn(num_steps_per_fit, num_episodes_per_fit, num_steps, num_episodes)


        dataset = self.agent.learn(environment=self.environment,
                                   verbose=verbose)
        return dataset


    def _reset(self, initial_state=None):
        pass



    def _step(self):
        pass