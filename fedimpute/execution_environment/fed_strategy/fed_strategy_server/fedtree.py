from copy import deepcopy
from typing import List, Tuple
from collections import OrderedDict
import numpy as np
from ...fed_strategy.fed_strategy_server.base_strategy import StrategyServer


class FedTreeStrategyServer(StrategyServer):

    def __init__(self, strategy_params):
        super(FedTreeStrategyServer, self).__init__('fedtree')
        self.strategy_params = strategy_params
        self.fine_tune_epochs = strategy_params.get('fine_tune_steps', 0)

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

        # federated tree sampling strategy
        sample_sizes = [item['sample_size'] for item in fit_res]
        sample_fracs = [size / sum(sample_sizes) for size in sample_sizes]

        np.random.seed(1203401)
        # all local trees
        global_trees = []
        for local_model_state_dict, sample_frac in zip(local_model_parameters, sample_fracs):
            local_trees = local_model_state_dict['estimators']
            sampled_trees = np.random.choice(local_trees, int(len(local_trees) * sample_frac), replace=False)
            global_trees.extend(sampled_trees)

        global_params = OrderedDict({"estimators": global_trees})
        # copy parameters for each client
        agg_model_parameters = [deepcopy(global_params) for _ in range(len(local_model_parameters))]
        agg_res = {}

        return agg_model_parameters, agg_res

    def fit_instruction(self, params_list: List[dict]) -> List[dict]:

        return [{'fit_model': True} for _ in range(len(params_list))]

    def update_instruction(self, params: dict) -> dict:

        return {}

