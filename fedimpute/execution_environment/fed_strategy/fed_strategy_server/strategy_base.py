from abc import ABC, abstractmethod
from typing import List, OrderedDict, Tuple, Union
import torch


class NNStrategyBaseServer(ABC):

    def __init__(self, name: str, initial_impute: str, fine_tune_epochs: int = 0):
        
        # basic parameters
        self.name: str = name
        self.initial_impute: str = initial_impute
        self.fine_tune_epochs: int = fine_tune_epochs

    @abstractmethod
    def initialization(self, global_model: torch.nn.Module, params: dict):
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def fit_instruction(self, params_list: List[dict]) -> List[dict]:
        pass

    @abstractmethod
    def update_instruction(self, params: dict) -> dict:
        return {'update_model': True}
    
    @abstractmethod
    def get_global_model_params(self) -> Union[OrderedDict, None]:
        pass
