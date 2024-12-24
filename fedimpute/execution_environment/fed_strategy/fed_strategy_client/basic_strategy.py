

class LocalStrategyClient:

    def __init__(self):
        self.name = 'local'
        self.description = 'Local'

class CentralStrategyClient:
    def __init__(self):
        self.name = 'central'
        self.description = 'Centralized'

class SimpleAvgStrategyClient:
    def __init__(self):
        self.name = 'simple_avg'
        self.description = 'Simple Averaging'

class FedTreeStrategyClient:

    def __init__(self):
        self.name = 'fedavg'
        self.description = 'Federated Tree'


