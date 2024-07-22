from typing import List, Tuple
import torch
from ...fed_strategy.fed_strategy_server import StrategyBaseServer
import copy


class ScaffoldStrategyServer(StrategyBaseServer):

    def __init__(self, server_learning_rate: float = 0.001, fine_tune_epochs: int = 0):

        super(ScaffoldStrategyServer, self).__init__('scaffold', 'fedavg', fine_tune_epochs)
        self.initial_impute = 'fedavg'
        self.fine_tune_epochs = 0
        self.server_learning_rate = server_learning_rate
        self.global_model = None
        self.global_c = None

    def initialization(self, global_model, params: dict):
        """
        Initialize the server
        :param global_model: global model
        :param params: parameters of initialization
        :return: None
        """
        self.global_model = global_model
        self.global_c = [torch.zeros_like(param) for param in global_model.parameters()]

    def aggregate_parameters(
            self, local_models: List[torch.nn.Module], fit_res: List[dict], params: dict, *args, **kwargs
    ) -> Tuple[List[torch.nn.Module], dict]:
        """
        Aggregate local models
        :param local_models: List of local model objects
        :param fit_res: List of fit results of local training
            - sample_size: int - number of samples used for training
        :param params: dictionary for information
        :param args: other params list
        :param kwargs: other params dict
        :return: List of aggregated model parameters, dict of aggregated results
        """

        global_model = copy.deepcopy(self.global_model)
        global_c = copy.deepcopy(self.global_c)
        num_clients = len(local_models)
        weights = torch.tensor([fit_res[cid]['sample_size'] for cid in range(num_clients)])
        weights = weights / weights.sum()
        for cid in range(num_clients):
            dy, dc = fit_res[cid]['delta_y'], fit_res[cid]['delta_c']
            for server_param, client_param in zip(global_model.parameters(), dy):
                server_param.data += client_param.data.clone() * weights[cid] * self.server_learning_rate
            for server_param, client_param in zip(global_c, dc):
                server_param.data += client_param.data.clone() * weights[cid]

        self.global_model = global_model
        self.global_c = global_c

        return [global_model for _ in range(num_clients)], {}

    def fit_instruction(self, params_list: List[dict]) -> List[dict]:

        return [{'fit_model': True} for _ in range(len(params_list))]

    def update_instruction(self, params: dict) -> dict:

        return {}