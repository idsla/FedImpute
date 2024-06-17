from copy import deepcopy
from typing import List, Tuple
from collections import OrderedDict
from ...fed_strategy.fed_strategy_server.base_strategy import StrategyServer


class CentralStrategyServer(StrategyServer):

    def __init__(self):
        super(CentralStrategyServer, self).__init__('central', 'central')
        self.intial_impute = 'central'
        self.fine_tune_epochs = 0

    def aggregate_parameters(
            self, local_model_parameters: List[OrderedDict], fit_res: List[dict], params: dict, *args, **kwargs
    ) -> Tuple[List[OrderedDict], dict]:
        """
        Aggregate local models
        :param local_model_parameters: List of local model parameters
        :param fit_res: List of fit results of local training
            - sample_size: int - number of samples used for training
        :param params: dictionary for information
        :param args: other params list
        :param kwargs: other params dict
        :return: List of aggregated model parameters, dict of aggregated results
        """
        central_model_params = local_model_parameters[-1]

        agg_model_parameters = [deepcopy(central_model_params) for _ in range(len(local_model_parameters))]
        agg_res = {}

        return agg_model_parameters, agg_res

    def fit_instruction(self, params_list: List[dict]) -> List[dict]:
        fit_instructions = []
        for _ in range(len(params_list) - 1):
            fit_instructions.append({'fit_model': False})

        fit_instructions.append({'fit_model': True})

        return fit_instructions

    def update_instruction(self, params: dict) -> dict:

        return {}
