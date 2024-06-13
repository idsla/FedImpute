from abc import ABC, abstractmethod
from typing import List, OrderedDict, Tuple


class StrategyServer(ABC):

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
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
        pass

    @abstractmethod
    def fit_instruction(self, params_list: List[dict]) -> List[dict]:
        pass

    @abstractmethod
    def update_instruction(self, params: dict) -> dict:
        pass