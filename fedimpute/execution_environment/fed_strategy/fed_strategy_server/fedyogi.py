from typing import List, Tuple, Optional, Dict, Union, OrderedDict
import numpy as np
import torch

from .fedavg import fedavg
from .strategy_base import NNStrategyBaseServer
from ..utils import get_parameters


class FedYogiStrategyServer(NNStrategyBaseServer):

    def __init__(
            self, eta=0.1, eta_l=0.1, beta_1=0.9, beta_2=0.99, tau=1e-9, weight_option='sample_size'
    ):
        super().__init__('fedyogi', 'fedavg', 0)
        self.global_model = None
        self.name = 'fedyogi'
        self.initial_impute = 'fedavg'
        self.fine_tune_epochs = 0
        self.eta = eta
        self.eta_l = eta_l
        self.tau = tau
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.m_t = None
        self.v_t = None
        self.weight_option = weight_option

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

        # compute delta
        global_model_dict = get_parameters(self.global_model, trainable_only=True, return_type='numpy_dict')
        delta_t = {k: aggregated_model_dict[k] - global_model_dict[k] for k in global_model_dict.keys()}

        # update m_t
        if not self.m_t:
            self.m_t = {k: np.zeros_like(v) for k, v in delta_t.items()}

        for k, delta_t_param in delta_t.items():
            self.m_t[k] = np.multiply(self.beta_1, self.m_t[k]) + (1.0 - self.beta_1) * delta_t_param

        # update v_t
        if not self.v_t:
            self.v_t = {k: np.zeros_like(v) for k, v in delta_t.items()}

        for k, delta_t_param in delta_t.items():
            self.v_t[k] = (
                    self.v_t[k] - (1.0 - self.beta_2) * np.multiply(delta_t_param, delta_t_param) *
                    np.sign(self.v_t[k] - np.multiply(delta_t_param, delta_t_param))
            )

        # final global model params
        new_global_model_dict = {
            k: global_model_dict[k] + self.eta * self.m_t[k] / (np.sqrt(self.v_t[k]) + self.tau)
            for k in global_model_dict.keys()
        }

        # update global model
        self.global_model.load_state_dict({k: torch.from_numpy(v) for k, v in new_global_model_dict.items()})

        # copy parameters for each client
        agg_model_parameters = [new_global_model_dict for _ in range(len(local_model_parameters))]
        agg_res = {}

        return agg_model_parameters, agg_res

    def fit_instruction(self, params_list: List[dict]) -> List[dict]:

        return [{
            'fit_model': True,
            'global_model_dict': get_parameters(self.global_model, trainable_only=True, return_type='numpy_dict')
        } for _ in range(len(params_list))]

    def update_instruction(self, params: dict) -> dict:

        return {}
    
    def get_global_model_params(self) -> Union[OrderedDict, None]:
        return get_parameters(self.global_model, trainable_only=True, return_type='state_dict')
