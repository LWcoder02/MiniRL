


class ReplayBuffer():
    def __init__(self):
        self.init = False


    def sample(self, batch_size: int):
        ...


    def add(self, dataset):
        ...