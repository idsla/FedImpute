from copy import deepcopy
from typing import List, Tuple, Union
from collections import OrderedDict
from ...fed_strategy.fed_strategy_server.base_strategy import StrategyServer


class LocalStrategyServer(StrategyServer):

    def __init__(self):
        super(LocalStrategyServer, self).__init__('local', 'local')
        self.initial_impute = 'local'
        self.fine_tune_epochs = 0

    def aggregate_parameters(
            self, local_model_parameters: List[OrderedDict], fit_res: List[dict], params: dict, *args, **kwargs
    ) -> Tuple[List[Union[OrderedDict, None]], dict]:
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
        return [None for _ in range(len(local_model_parameters))], {}


    def fit_instruction(self, params_list: List[dict]) -> List[dict]:

        return [{'fit_model': True} for _ in range(len(params_list))]

    def update_instruction(self, params: dict) -> dict:
        return {}