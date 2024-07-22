from abc import ABC, abstractmethod
from typing import List, OrderedDict, Tuple
import torch


class StrategyBaseServer(ABC):

    def __init__(self, name: str, initial_impute: str, fine_tune_epochs: int = 0):
        self.name = name
        self.initial_impute = initial_impute
        self.fine_tune_epochs = fine_tune_epochs

    @abstractmethod
    def initialization(self, global_model: torch.nn.Module, params: dict):
        pass

    @abstractmethod
    def aggregate_parameters(
            self, local_models: List[torch.nn.Module], fit_res: List[dict], params: dict, *args, **kwargs
    ) -> Tuple[List[torch.nn.Module], dict]:
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