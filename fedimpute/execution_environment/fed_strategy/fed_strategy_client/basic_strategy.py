
class BasicStrategyClient:
    def __init__(self):
        pass

class LocalStrategyClient(BasicStrategyClient):

    def __init__(self):
        self.name = 'local'
        self.description = 'Local'

class CentralStrategyClient(BasicStrategyClient):
    def __init__(self):
        self.name = 'central'
        self.description = 'Centralized'

class SimpleAvgStrategyClient(BasicStrategyClient):
    def __init__(self):
        self.name = 'simple_avg'
        self.description = 'Simple Averaging'

class FedMeanStrategyClient(SimpleAvgStrategyClient):
    def __init__(self):
        self.name = 'fedmean'
        self.description = 'Federated Mean with simple averaging'

class FedMICEStrategyClient(SimpleAvgStrategyClient):
    def __init__(self):
        self.name = 'fedmice'
        self.description = 'Federated MICE with simple averaging'

class FedEMStrategyClient(SimpleAvgStrategyClient):
    def __init__(self):
        self.name = 'fedem'
        self.description = 'Federated EM with simple averaging'

class FedTreeStrategyClient(BasicStrategyClient):

    def __init__(self):
        self.name = 'fedavg'
        self.description = 'Federated Tree'


