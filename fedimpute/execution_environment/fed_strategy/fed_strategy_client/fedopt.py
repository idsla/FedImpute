from .fedavg import FedAvgStrategyClient

class FedAdamStrategyClient(FedAvgStrategyClient):

    def __init__(self):
        super().__init__(global_initialize = True)
        self.name = 'fedadam'

class FedAdagradStrategyClient(FedAvgStrategyClient):

    def __init__(self):
        super().__init__(global_initialize = True)
        self.name = 'fedadagrad'

class FedYogiStrategyClient(FedAvgStrategyClient):

    def __init__(self):
        super().__init__(global_initialize = True)
        self.name = 'fedyogi'
