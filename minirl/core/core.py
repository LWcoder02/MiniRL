from minirl.core.logic.run_logic import TrainLogic
from minirl.core.agent import Agent
from minirl.core.dataset import Dataset
from minirl.core.environment import Environment


class Core():
    def __init__(self, agent: Agent, environment: Environment):
        self.agent: Agent = agent
        self.environment: Environment = environment

        self._logic: TrainLogic = TrainLogic()

        self._state = None




    def learn(self, num_steps: int = None, num_episodes: int = None,
              num_steps_per_fit: int = None, num_episodes_per_fit: int= None, quiet: bool = False):
        self._logic.init_learn(num_steps_per_fit=num_steps_per_fit, num_episodes_per_fit=num_episodes_per_fit)

        dataset = Dataset()

        self._run_impl(dataset, num_steps=num_steps, num_episodes=num_episodes)


    def evaluate(self, num_steps: int = None, num_episodes: int = None, quiet: bool = False):
        self._logic.init_evaluate()

        dataset = Dataset()

        return self._run_impl(dataset, num_steps=num_steps, num_episodes=num_episodes)


    def _run_impl(self, dataset: Dataset, num_steps: int, num_episodes: int, initial_state=None):
        self._logic.init_run(num_steps=num_steps, num_episodes=num_episodes)

        done = True
        while self._logic.move_condition():
            if done:
                self._reset(initial_state=initial_state)

            sample, info = self._step()
            self._logic.after_step(done=done)

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


    def _reset(self, initial_state) -> None:
        state, info = self.environment.reset(initial_state=initial_state)
        self._state = state