from minirl.core.logic.run_logic import TrainLogic
from minirl.core.agent import Agent
from minirl.core.dataset import Dataset
from minirl.core.environment import Environment


class Core(object):
    def __init__(self, agent: Agent, environment: Environment):
        self.agent: Agent = agent
        self.environment: Environment = environment