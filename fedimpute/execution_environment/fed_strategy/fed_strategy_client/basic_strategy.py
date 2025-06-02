from .strategy_base import StrategyBaseClient
import torch
import numpy as np
from typing import Tuple
from ...imputation.base import BaseNNImputer

class BasicStrategyClient(StrategyBaseClient):
    def __init__(self):
        pass

    def set_parameters(self, updated_model_params: dict, local_model: torch.nn.Module, params: dict):
        """
        Set parameters global model to local model
        :param updated_model_params: received updated model parameters
        :param local_model: local model
        :param params: config params
        :return: None
        """
        pass

    def get_parameters(self, local_model: torch.nn.Module, params: dict) -> dict:
        """
        Get parameters from local model
        :param local_model: local model
        :param params: config params
        :return: parameters dict
        """
        pass

    def pre_training_setup(self, params: dict) -> dict:
        """
        Local training pre setup
        :param params: config params
        :return: setup results dict
        """
        pass

    def post_training_setup(self, params: dict) -> dict:
        """
        Local training post setup
        :param params: config params
        :return: setup results dict
        """
        pass

    def train_local_nn_model(
            self, imputer: BaseNNImputer, training_params: dict, X_train_imp: np.ndarray,
            y_train: np.ndarray, X_train_mask: np.ndarray
    ) -> Tuple[dict, dict]:
        """
        Train local nn model
        :param imputer: imputer
        :param training_params: training params
        :param X_train_imp: Imputed training data
        :param y_train: y training data
        :param X_train_mask: mask data
        :return: local model and training results dict
        """
        pass

    def get_fit_res(self, local_model: torch.nn.Module, params: dict) -> dict:
        pass

class LocalStrategyClient(BasicStrategyClient):

    def __init__(self):
        self.name = 'local'
        self.description = 'Local'
        
    def __str__(self):
        return f"Local Strategy Client"

    def __repr__(self):
        return f"Local Strategy Client"

class CentralStrategyClient(BasicStrategyClient):
    def __init__(self):
        self.name = 'central'
        self.description = 'Centralized'
    
    def __str__(self):
        return f"Central Strategy Client"

    def __repr__(self):
        return f"Central Strategy Client"

class SimpleAvgStrategyClient(BasicStrategyClient):
    def __init__(self):
        self.name = 'simple_avg'
        self.description = 'Simple Averaging'
    
    def __str__(self):
        return f"Simple Average Strategy Client"

    def __repr__(self):
        return f"Simple Average Strategy Client"

class FedMeanStrategyClient(SimpleAvgStrategyClient):
    def __init__(self):
        self.name = 'fedmean'
        self.description = 'Federated Mean with simple averaging'

    def __str__(self):
        return f"Federated Mean Strategy Client"

    def __repr__(self):
        return f"Federated Mean Strategy Client"

class FedMICEStrategyClient(SimpleAvgStrategyClient):
    def __init__(self):
        self.name = 'fedmice'
        self.description = 'Federated MICE with simple averaging'
    
    def __str__(self):
        return f"Federated MICE Strategy Client"

    def __repr__(self):
        return f"Federated MICE Strategy Client"

class FedEMStrategyClient(SimpleAvgStrategyClient):
    def __init__(self):
        self.name = 'fedem'
        self.description = 'Federated EM with simple averaging'

    def __str__(self):
        return f"Federated EM Strategy Client"

    def __repr__(self):
        return f"Federated EM Strategy Client"

class FedTreeStrategyClient(BasicStrategyClient):

    def __init__(self):
        self.name = 'fedavg'
        self.description = 'Federated Tree'

    def __str__(self):
        return f"Federated Tree Strategy Client"

    def __repr__(self):
        return f"Federated Tree Strategy Client"


