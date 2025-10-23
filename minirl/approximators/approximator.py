from minirl.core.serialization import Serialization

class Approximator(Serialization):
    
    def __init__(self):
        ...


    def predict(self, *args):
        raise NotImplementedError()