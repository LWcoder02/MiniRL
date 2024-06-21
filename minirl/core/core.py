from MiniRL.minirl.core.logic.run_logic import TrainLogic
from minirl.core.agent import Agent


class Core():
    def __init__(self, agent: Agent, environment):
        self.agent = agent
        self.environment = environment

        self._logic = TrainLogic()


    def train(self, config, num_steps=None, num_steps_fit=None,
              num_episodes=None, num_episodes_fit=None, verbose=True):
        

        scores = {
            'J': []
        }
        # Start the training
        for epoch in config['epochs']:
            self.agent.update_policy() # i.e. set epsilon
            print(f"Beginn learning of epoch {epoch+1}")
            self._logic.init_learn(num_steps_fit, num_episodes_fit, num_steps, num_episodes)
            self.agent.learn(environment=self.environment, verbose=verbose)

            print(f"Beginn evaluation of epoch {epoch+1}")
            self._logic.init_evaluate(num_steps, num_episodes)
            dataset = self.agent.evaluate(verbose=verbose)
            scores['J'].append(self.compute_J(dataset))


    def _reset(self, initial_state=None):
        pass


    def compute_J(self, dataset):
        pass