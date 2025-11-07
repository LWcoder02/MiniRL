from minirl.core.agent import Agent
from minirl.core.environment import Environment
from minirl.core.dataset import Dataset

from typing import Tuple, Any, Dict

AgentID = str | int
ObsType = Any
ActionTpe = Any
ObsDict = Dict[AgentID, ObsType]
ActionDict = Dict[ObsType, int]


class SingleAgentCore(object):
    def __init__(self, agent: Agent, environment: Environment,
                 dataset_type: Dataset):
        self._agent: Agent = agent
        self._environment: Environment = environment
        self._dataset = dataset_type


    def learn(self):
        # dataset = self._dataset.generate(environment_info=info, num_steps=5)
        # return dataset
        raise NotImplementedError()
    

    def evaluate(self):
        # dataset = self._dataset.generate(info, num_steps=5)
        # return dataset
        raise NotImplementedError()
    

    def run(self):
        # run learning

        # run evaluation
        raise NotImplementedError()
    

    def _run_imple(self):
        raise NotImplementedError()