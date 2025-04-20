from .fedavg import FedAvgStrategyClient

class LocalNNStrategyClient(FedAvgStrategyClient):

    def __init__(self):
        super().__init__(global_initialize = False)
        self.name = 'local_nn'


class CentralNNStrategyClient(FedAvgStrategyClient):

    def __init__(self):
        super().__init__(global_initialize = False)
        self.name = 'central_nn'

