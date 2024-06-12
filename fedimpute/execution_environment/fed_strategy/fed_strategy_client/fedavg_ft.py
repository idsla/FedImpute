from ...fed_strategy.fed_strategy_client import StrategyClient
import torch
from typing import Tuple


class FedAvgFtStrategyClient(StrategyClient):

    def __init__(self, strategy_params):
        self.strategy_params = strategy_params
        super().__init__('fedavg_ft')

    def pre_training_setup(self, model: torch.nn.Module, params: dict):
        if 'freeze_encoder' in params and params['freeze_encoder']:
            for param in model.encoder.parameters():
                param.requires_grad = False

    def fed_updates(self, model: torch.nn.Module):
        pass

    def post_training_setup(self, model: torch.nn.Module):
        pass
