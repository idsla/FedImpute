from .fedavg import FedAvgStrategyClient

class LocalNNStrategyClient(FedAvgStrategyClient):

    def __init__(self):
        super().__init__(global_initialize = False)
        self.name = 'local_nn'

    def __str__(self):
        return f"LocalNNStrategyClient"

    def __repr__(self):
        return f"LocalNNStrategyClient"



class CentralNNStrategyClient(FedAvgStrategyClient):

    def __init__(self):
        super().__init__(global_initialize = False)
        self.name = 'central_nn'

    def __str__(self):
        return f"CentralNNStrategyClient"

    def __repr__(self):
        return f"CentralNNStrategyClient"   


