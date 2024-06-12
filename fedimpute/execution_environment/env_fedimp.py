class FedImputationEnvironment:

    def __init__(self, params):
        self.params = params
        self.env = None
        self.simulator = None
        self.evaluator = None
        self.benchmark = None

    def setup_env(self):
        pass

    def setup_env_from_simulator(self, simulator):
        pass

    def run_federated_imputation(self):
        pass

    def save_env(self):
        pass

    def load_env(self):
        pass

    def reset_env(self):
        pass
