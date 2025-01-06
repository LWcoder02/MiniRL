from minirl.core.serialization import Serialization
from minirl.core.logic.run_logic import TrainLogic
from tqdm import tqdm
from minirl.core.dataset import Dataset


class Agent(Serialization):
    def __init__(self, environment, policy, environment_info):
        self.policy = policy
        self.environment = environment
        self.env_info = environment_info

        self._logic = TrainLogic()

        self._state = None



    def _train(self, dataset):
        raise NotImplementedError("Train method is not implemented since Agent is an abstract class")



    def _run_impl(self, dataset):
        self._logic.init_run()

        done = True
        while self._logic.move_condition():
            if done:
                self._reset()


            sample = self._step()

            dataset.append(sample)


            if self._logic.fit_condition():
                self._train(dataset)

                dataset.clear()

            
            done = sample[4]


        return dataset
    

    def _step(self):
        action = self.draw_action(self._state)
        next_state, reward, done, info = self.environment.step(action)

        state = self._state
        self._state = next_state

        return (state, action, reward, next_state, done), info
    

    def _reset(self):
        pass


    def learn(self, num_steps=None, num_episodes=None,
              num_steps_per_fit=None, num_episodes_per_fit=None, quiet=False):
        self._logic.init_learn(num_steps_per_fit=num_steps_per_fit, num_episodes_per_fit=num_episodes_per_fit)

        dataset = Dataset()

        self._run_impl(dataset)


    def evaluate(self, num_steps=None, num_episodes=None, quiet=False):
        self._logic.init_evaluate()

        dataset = Dataset()

        return self._run_impl(dataset)


    def draw_action(self, state):
        return self.policy.draw_action(state)

