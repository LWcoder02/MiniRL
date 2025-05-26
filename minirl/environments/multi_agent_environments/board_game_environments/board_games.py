from minirl.core.environment import Environment


class AbstractBoardGame(Environment):
    def __init__(self, environment_info):
        super().__init__(environment_info=environment_info)


    def get_legal_actions(self):
        raise NotImplementedError