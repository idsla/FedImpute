from typing import List, Tuple, Union, OrderedDict
import numpy as np
from copy import deepcopy

from ...fed_strategy.fed_strategy_server.strategy_base import NNStrategyBaseServer
from ..utils import get_parameters, convert_params_format

class CentralNNStrategyServer(NNStrategyBaseServer):

    def __init__(self, fine_tune_epochs: int = 0):
        super().__init__('central_nn', 'central', fine_tune_epochs)
        self.initial_impute = 'central'
        self.global_model_params_dict: OrderedDict = None

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
        central_model_params = local_model_parameters[-1]

        agg_model_parameters = [deepcopy(central_model_params) for _ in range(len(local_model_parameters))]
        agg_res = {}
        
        # update global model parameters
        self.global_model_params_dict = central_model_params

        return agg_model_parameters, agg_res

    def initialization(self, global_model, params: dict):
        """
        Initialize the server
        :param global_model: global model
        :param params: parameters of initialization
        :return: None
        """
        pass

    def fit_instruction(self, params_list: List[dict]) -> List[dict]:

        fit_instructions = []
        for _ in range(len(params_list) - 1):
            fit_instructions.append({'fit_model': False})

        fit_instructions.append({'fit_model': True})

        return fit_instructions


    def update_instruction(self, params: dict) -> dict:

        return {}
    
    def get_global_model_params(self) -> Union[OrderedDict, None]:
        return convert_params_format(self.global_model_params_dict, output_type='state_dict')
    
    def __str__(self):
        return f"CentralNNStrategyServer"

    def __repr__(self):
        return f"CentralNNStrategyServer"
