from ...fed_strategy.fed_strategy_client import StrategyClient
import torch
from typing import Tuple


class FedAvgStrategyClient(StrategyClient):

    def __init__(self, strategy_params):
        self.strategy_params = strategy_params
        super().__init__('fedavg')

    def pre_training_setup(self, model: torch.nn.Module, params: dict):
        pass

    def fed_updates(self, model: torch.nn.Module):
        pass

    def post_training_setup(self, model: torch.nn.Module):
        pass
