from typing import List, Tuple, Union
from typing import List, OrderedDict, Tuple, Union
import numpy as np

from ...fed_strategy.fed_strategy_server.strategy_base import NNStrategyBaseServer

class LocalNNStrategyServer(NNStrategyBaseServer):

    def __init__(self, fine_tune_epochs: int = 0):
        super().__init__('local_nn', 'local', fine_tune_epochs)
        self.initial_impute = 'local'

    def aggregate_parameters(
            self, local_model_parameters: List[dict], fit_res: List[dict], params: dict, *args, **kwargs
    ) -> Tuple[List[dict], dict]:
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
        agg_res = {}

        return local_model_parameters, agg_res

    def initialization(self, global_model, params: dict):
        """
        Initialize the server
        :param global_model: global model
        :param params: parameters of initialization
        :return: None
        """
        pass

    def fit_instruction(self, params_list: List[dict]) -> List[dict]:

        return [{'fit_model': True} for _ in range(len(params_list))]

    def update_instruction(self, params: dict) -> dict:

        return {}
    
    def get_global_model_params(self) -> Union[OrderedDict, None]:
        return None
