from .fedavg import FedAvgStrategyClient

class FedAdamStrategyClient(FedAvgStrategyClient):

    def __init__(self):
        super().__init__(global_initialize = True)
        self.name = 'fedadam'
        
    def __str__(self):
        return f"FedAdam Strategy Client"

    def __repr__(self):
        return f"FedAdam Strategy Client"

class FedAdagradStrategyClient(FedAvgStrategyClient):

    def __init__(self):
        super().__init__(global_initialize = True)
        self.name = 'fedadagrad'
    
    def __str__(self):
        return f"FedAdagrad Strategy Client"

    def __repr__(self):
        return f"FedAdagrad Strategy Client"

class FedYogiStrategyClient(FedAvgStrategyClient):

    def __init__(self):
        super().__init__(global_initialize = True)
        self.name = 'fedyogi'
    
    def __str__(self):
        return f"FedYogi Strategy Client"

    def __repr__(self):
        return f"FedYogi Strategy Client"
