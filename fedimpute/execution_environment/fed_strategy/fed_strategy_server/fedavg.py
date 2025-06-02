from typing import List, Tuple, Union, OrderedDict
import numpy as np

from ...fed_strategy.fed_strategy_server.strategy_base import NNStrategyBaseServer
from ..utils import get_parameters, convert_params_format

def fedavg(
    local_model_parameters: List[dict], fit_res: List[dict], weight_option: str = 'sample_size'
) -> dict:

    if len(local_model_parameters) == 0:
        raise ValueError("No local model parameters found")
    else:
        averaged_model_state_dict = {
            key: np.zeros_like(value) for key, value in local_model_parameters[0].items()
        }
        if weight_option == 'sample_size':
            if not all('sample_size' in item for item in fit_res):
                raise ValueError("sample_size not found in fit_res")
            sample_sizes = [item['sample_size'] for item in fit_res]
            normalized_coefficient = [size / sum(sample_sizes) for size in sample_sizes]
        elif weight_option == 'num_clients':
            num_clients = len(local_model_parameters)
            normalized_coefficient = [1 / num_clients for _ in range(num_clients)]
        else:
            raise ValueError("weight_option must be either 'sample_size' or 'num_clients'")

        for it, local_model_state_dict in enumerate(local_model_parameters):
            for key in local_model_state_dict.keys():
                if it == 0:
                    averaged_model_state_dict[key] = normalized_coefficient[it] * local_model_state_dict[key]
                else:
                    averaged_model_state_dict[key] += normalized_coefficient[it] * local_model_state_dict[key]

        return averaged_model_state_dict


class FedAvgStrategyServer(NNStrategyBaseServer):

    def __init__(self, fine_tune_epochs: int = 0, weight_option: str = 'sample_size'):
        super().__init__('fedavg', 'fedavg', fine_tune_epochs)
        self.global_model = None
        self.initial_impute = 'fedavg'
        self.weight_option = weight_option
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
    
        # federated averaging implementation
        aggregated_model_dict = fedavg(local_model_parameters, fit_res, self.weight_option)
        self.global_model.load_state_dict(convert_params_format(aggregated_model_dict, output_type='state_dict'))

        # copy parameters for each client
        agg_model_parameters = [aggregated_model_dict for _ in range(len(local_model_parameters))]
        agg_res = {}
        
        # update global model parameters
        self.global_model_params_dict = aggregated_model_dict

        return agg_model_parameters, agg_res

    def initialization(self, global_model, params: dict):
        """
        Initialize the server
        :param global_model: global model
        :param params: parameters of initialization
        :return: None
        """
        self.global_model = global_model
        # self.global_c = [torch.zeros_like(param) for param in global_model.parameters()]
        pass

    def fit_instruction(self, params_list: List[dict]) -> List[dict]:

        return [{
            'fit_model': True,
            'global_model_dict': get_parameters(self.global_model, trainable_only=True, return_type='numpy_dict')
        } for _ in range(len(params_list))]

    def update_instruction(self, params: dict) -> dict:

        return {}
    
    def get_global_model_params(self) -> Union[OrderedDict, None]:
        return convert_params_format(self.global_model_params_dict, output_type='state_dict')
    
    def __str__(self):
        return f"FedAvg Strategy Server"

    def __repr__(self):
        return f"FedAvg Strategy Server"
