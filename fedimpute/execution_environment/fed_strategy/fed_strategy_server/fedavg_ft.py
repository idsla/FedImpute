from typing import List, Tuple
import torch

from ...fed_strategy.fed_strategy_server.fedavg import FedAvgStrategyServer

class FedAvgFtStrategyServer(FedAvgStrategyServer):

    def __init__(self, fine_tune_epochs: int = 100):
        super(FedAvgFtStrategyServer, self).__init__(fine_tune_epochs)

# from copy import deepcopy
# from typing import List, Tuple
# from collections import OrderedDict
# import torch
#
# from ...fed_strategy.fed_strategy_server.strategy_base import StrategyBaseServer
#
#
# class FedAvgFtStrategyBaseServer(StrategyBaseServer):
#
#     def __init__(self, fine_tune_epochs: int = 200):
#         super(FedAvgFtStrategyBaseServer, self).__init__('fedavg_ft', 'fedavg')
#         self.initial_impute = 'fedavg'
#         self.fine_tune_epochs = fine_tune_epochs
#
#     def initialization(self, global_model, params: dict):
#         """
#         Initialize the server
#         :param global_model: global model
#         :param params: parameters of initialization
#         :return: None
#         """
#         self.global_model = global_model
#         self.global_c = [torch.zeros_like(param) for param in global_model.parameters()]
#
#     def aggregate_parameters(
#             self, local_models: List[torch.nn.Module], fit_res: List[dict], params: dict, *args, **kwargs
#     ) -> Tuple[List[torch.nn.Module], dict]:
#         """
#         Aggregate local models
#         :param local_models: List of local model objects
#         :param fit_res: List of fit results of local training
#             - sample_size: int - number of samples used for training
#         :param params: dictionary for information
#         :param args: other params list
#         :param kwargs: other params dict
#         :return: List of aggregated model parameters, dict of aggregated results
#         """
#         # clear the server model
#         for server_params in self.global_model.parameters():
#             server_params.data = torch.zeros_like(server_params.data)
#
#         # federated averaging
#         weights = torch.tensor([fit_res[cid]['sample_size'] for cid in range(len(local_models))])
#         weights = weights / weights.sum()
#         for cid in range(len(local_models)):
#             for server_param, client_param in zip(self.global_model.parameters(), local_models[cid].parameters()):
#                 server_param.data += client_param.data.clone() * weights[cid]
#
#         return [self.global_model for _ in range(len(local_models))], {}
#
#     def fit_instruction(self, params_list: List[dict]) -> List[dict]:
#
#         return [{'fit_model': True} for _ in range(len(params_list))]
#
#     def update_instruction(self, params: dict) -> dict:
#         return {'update_model': True}
