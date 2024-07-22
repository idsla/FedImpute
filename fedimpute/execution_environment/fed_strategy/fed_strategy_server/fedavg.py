from typing import List, Tuple
import torch

from ...fed_strategy.fed_strategy_server.strategy_base import StrategyBaseServer


class FedAvgStrategyServer(StrategyBaseServer):

    def __init__(self, fine_tune_epochs: int = 0):
        super(FedAvgStrategyServer, self).__init__('fedavg', 'fedavg', fine_tune_epochs)
        self.initial_impute = 'fedavg'

    # def aggregate_parameters(
    #         self, local_model_parameters: List[OrderedDict], fit_res: List[dict], params: dict, *args, **kwargs
    # ) -> Tuple[List[OrderedDict], dict]:
    #     """
    #     Aggregate local models
    #     :param local_model_parameters: List of local model parameters
    #     :param fit_res: List of fit results of local training
    #         - sample_size: int - number of samples used for training
    #     :param params: dictionary for information
    #     :param args: other params list
    #     :param kwargs: other params dict
    #     :return: List of aggregated model parameters, dict of aggregated results
    #     """
    #
    #     # federated averaging implementation
    #     averaged_model_state_dict = OrderedDict([])  # global parameters
    #     sample_sizes = [item['sample_size'] for item in fit_res]
    #     normalized_coefficient = [size / sum(sample_sizes) for size in sample_sizes]
    #
    #     for it, local_model_state_dict in enumerate(local_model_parameters):
    #         for key in local_model_state_dict.keys():
    #             if it == 0:
    #                 averaged_model_state_dict[key] = normalized_coefficient[it] * local_model_state_dict[key]
    #             else:
    #                 averaged_model_state_dict[key] += normalized_coefficient[it] * local_model_state_dict[key]
    #
    #     # copy parameters for each client
    #     agg_model_parameters = [deepcopy(averaged_model_state_dict) for _ in range(len(local_model_parameters))]
    #     agg_res = {}
    #
    #     return agg_model_parameters, agg_res
    #
    # def fit_instruction(self, params_list: List[dict]) -> List[dict]:
    #
    #     return [{'fit_model': True} for _ in range(len(params_list))]
    #
    # def update_instruction(self, params: dict) -> dict:
    #
    #     return {}
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
        # clear the server model
        for server_params in self.global_model.parameters():
            server_params.data = torch.zeros_like(server_params.data)

        # federated averaging
        weights = torch.tensor([fit_res[cid]['sample_size'] for cid in range(len(local_models))])
        weights = weights / weights.sum()
        for cid in range(len(local_models)):
            for server_param, client_param in zip(self.global_model.parameters(), local_models[cid].parameters()):
                server_param.data += client_param.data.clone() * weights[cid]

        return [self.global_model for _ in range(len(local_models))], {}

    def fit_instruction(self, params_list: List[dict]) -> List[dict]:

        return [{'fit_model': True} for _ in range(len(params_list))]

    def update_instruction(self, params: dict) -> dict:

        return {}