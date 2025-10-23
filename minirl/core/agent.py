from minirl.core.serialization import Serialization
from minirl.core.logic.run_logic import TrainLogic
from tqdm import tqdm
from minirl.core.dataset import Dataset
from minirl.core.environment import Environment, EnvironmentInfo
from minirl.policy.policy import Policy
from minirl.core.backend import Backend

class AgentInfo(Serialization):
    def __init__(self, backend):
        self.backend = backend
        self.device = 'cpu'


class Agent(Serialization):
    def __init__(self, environment: Environment, policy: Policy, backend: str = 'numpy'):
        self.policy: Policy = policy
        self.environment: Environment = environment
        self.env_info: EnvironmentInfo = self.environment.get_environment_info()
        self._agent_info = AgentInfo(backend=backend)

        self._logic = TrainLogic()

        self._backend: Backend = Backend.get_backend(backend)

        self._state = None



    def train(self, dataset):
        raise NotImplementedError("Train method is not implemented since Agent is an abstract class")



    def _run_impl(self, dataset: Dataset, num_steps, num_episodes, initial_state=None):
        self._logic.init_run(num_steps, num_episodes)

        done = True

        while self._logic.move_condition():
            if done:
                self._reset(initial_state=initial_state)


            state, action, reward, next_state, done, _ = self._step()
            self._logic.after_step(done)
            dataset.append((state, action, reward, next_state, done))

            if self._logic.fit_condition():
                self.train(dataset)

                dataset.clear()

        return dataset
    

    def _step(self):
        action = self.draw_action(self._state)
        next_state, reward, done, info = self.environment.step(action)

        state = self._state
        self._state = next_state

        return state, action, reward, next_state, done, info
    

    def _reset(self, initial_state):
        # initial_state = self._logic.get_initial_state(initial_state)

        state, info = self.environment.reset(initial_state=initial_state)
        self._state = state


    def learn(self, num_steps=None, num_episodes=None,
              num_steps_per_fit=None, num_episodes_per_fit=None, quiet=False):
        
        self._logic.init_learn(num_steps_per_fit=num_steps_per_fit, num_episodes_per_fit=num_episodes_per_fit)

        dataset = Dataset.generate(environment_info=self.env_info, agent_info=self._agent_info,
                                   num_steps=num_steps_per_fit, num_episodes=num_episodes_per_fit)
        self._run_impl(dataset, num_steps=num_steps, num_episodes=num_episodes)


    def evaluate(self, num_steps=None, num_episodes=None, quiet=False):
        self._logic.init_evaluate()

        dataset = Dataset()

        return self._run_impl(dataset, num_steps=num_steps, num_episodes=num_episodes)


    def draw_action(self, state):
        return self.policy.draw_action(state)

