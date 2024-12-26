from typing import List, Tuple
import torch
from ...fed_strategy.fed_strategy_server import NNStrategyBaseServer
import copy
from ..utils import get_parameters


class ScaffoldStrategyServer(NNStrategyBaseServer):

    def __init__(self, server_learning_rate: float = 1.0, fine_tune_epochs: int = 0):

        super(ScaffoldStrategyServer, self).__init__('scaffold', 'fedavg', fine_tune_epochs)
        self.initial_impute = 'fedavg'
        self.fine_tune_epochs = 0
        self.server_learning_rate = server_learning_rate
        self.global_model = None
        self.global_c = None

    @staticmethod
    def get_parameters(local_model: torch.nn.Module, params: dict) -> dict:
        return get_parameters(local_model, trainable_only=True, return_type='numpy_dict')

    def initialization(self, global_model, params: dict):
        """
        Initialize the server
        :param global_model: global model
        :param params: parameters of initialization
        :return: None
        """
        self.global_model = global_model
        self.global_c = [
            torch.zeros_like(param) for param in
            get_parameters(global_model, trainable_only=True, return_type='parameters')
        ]

    def aggregate_parameters(
        self, local_model_parameters: List[dict], fit_res: List[dict], params: dict, *args, **kwargs
    ) -> Tuple[List[dict], dict]:
        """
        Aggregate local models
        :param local_model_parameters: List of local model numpy dict
        :param fit_res: List of fit results of local training
            - sample_size: int - number of samples used for training
        :param params: dictionary for information
        :param args: other params list
        :param kwargs: other params dict
        :return: List of aggregated model parameters, dict of aggregated results
        """

        global_model = copy.deepcopy(self.global_model)
        global_c = copy.deepcopy(self.global_c)
        num_clients = len(local_model_parameters)
        weights = torch.tensor([fit_res[cid]['sample_size'] for cid in range(num_clients)])
        weights = weights / weights.sum()
        for cid in range(num_clients):
            dy, dc = fit_res[cid]['delta_y'], fit_res[cid]['delta_c']
            for server_param, client_param in zip(get_parameters(global_model, trainable_only=True, return_type='parameters', copy = False), dy):
                server_param.data += client_param.data.clone() * weights[cid] * self.server_learning_rate
            for server_param, client_param in zip(global_c, dc):
                server_param.data += client_param.data.clone() * weights[cid]

        self.global_model = global_model
        self.global_c = global_c

        return [self.get_parameters(self.global_model, {}) for _ in range(num_clients)], {}

    def fit_instruction(self, params_list: List[dict]) -> List[dict]:

        return [
            {
                'fit_model': True,
                'global_c': self.global_c,
                'global_model_dict': get_parameters(
                    self.global_model, trainable_only=True, return_type='numpy_dict')
            } for _ in range(len(params_list))
        ]

    def update_instruction(self, params: dict) -> dict:

        return {}