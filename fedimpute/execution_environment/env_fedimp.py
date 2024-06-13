from typing import List

import numpy as np

from .loaders.load_environment import setup_clients, setup_server


class FedImputationEnvironment:

    def __init__(self, params):

        self.params = params

        self.clients = None
        self.server = None
        self.workflow = None

        self.env = None
        self.simulator = None
        self.evaluator = None
        self.benchmark = None

    def setup_env(
        self, clients_train_data: List[np.ndarray], clients_test_data: List[np.ndarray],
        clients_ms_data: List[np.ndarray], clients_seeds: List[int], data_config: dict
    ):
        clients_data = list(zip(clients_train_data, clients_test_data, clients_ms_data))
        self.clients = setup_clients(clients_data, clients_seeds, data_config)
        self.server = setup_server()

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
