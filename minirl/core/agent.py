from minirl.core.serialization import Serialization


class Agent(Serialization):
    def __init__(self, env_info, policy):
        self.policy = policy
        self.env_info = env_info



    def train(self, dataset):
        raise NotImplementedError("Train method is not implemented since Agent is an abstract class")


    def draw_action(self, state):
        return self.polcy.draw_action(state)

