from ...fed_strategy.fed_strategy_client.base_strategy import StrategyClient, fit_local_model_base
import torch
from typing import Tuple


class LocalStrategyClient(StrategyClient):

    def __init__(self):
        super().__init__('local')

    def pre_training_setup(self, model: torch.nn.Module, params: dict):
        pass

    def fed_updates(self, model: torch.nn.Module):
        pass

    def post_training_setup(self, model: torch.nn.Module):
        pass
