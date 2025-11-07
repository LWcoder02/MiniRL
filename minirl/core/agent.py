from minirl.core.serialization import Serialization
from minirl.core.environment import EnvironmentInfo
from minirl.policy.policy import Policy
from minirl.core.backend import Backend


class AgentInfo(Serialization):
    def __init__(self, backend: Backend, agent_id: int | str, device: str = "cpu"):
        self.backend = backend
        self.device = device
        self.agent_id = agent_id



class Agent(Serialization):
    def __init__(self,
                 environment_info: EnvironmentInfo,
                 policy: Policy,
                 agent_id: int | str = "agent_0",
                 backend: str = 'numpy'):
        self._policy: Policy = policy
        self._env_info: EnvironmentInfo = environment_info
        self._agent_id = agent_id
        self._agent_info = AgentInfo(backend=backend, agent_id=agent_id)

        self._agent_backend: Backend = Backend.get_backend(backend)
        self._env_backend: Backend = Backend.get_backend(self._env_info.backend)



    def train(self, dataset):
        raise NotImplementedError("Train method is not implemented since Agent is an abstract class")


    def draw_action(self, state):
        state = self._convert_to_agent_backend(array=state)
        action = self._policy.draw_action(state)
        return self._convert_to_env_backend(action)


    def _convert_to_env_backend(self, array):
        return self._env_backend.convert_to_backend(self._agent_backend, array)


    def _convert_to_agent_backend(self, array):
        return self._agent_backend.convert_to_backend(self._env_backend, array)
    

    @property
    def info(self):
        return self._agent_info