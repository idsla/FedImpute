from copy import deepcopy
from typing import List, Union

import numpy as np

from .loaders.load_environment import setup_clients, setup_server
from .loaders.load_workflow import load_workflow
from .utils.evaluator import Evaluator
from .utils.result_analyzer import ResultAnalyzer
from .utils.tracker import Tracker
from fedimpute.simulator import Simulator


class FedImputeEnv:

    def __init__(self):

        # clients, server and workflow
        self.clients = None
        self.server = None
        self.workflow = None

        # imputer and fed strategy
        self.imputer_name = None
        self.fed_strategy_name = None
        self.workflow_name = None
        self.imputer_params = {}
        self.fed_strategy_params = {}
        self.workflow_params = {}

        # other components
        self.simulator = None
        self.evaluator = None
        self.tracker = None
        self.result_analyzer = None
        self.benchmark = None
        self.env_dir_path = None

    def configuration(
            self, imputer: str, fed_strategy: Union[str, None] = None,
            imputer_params: Union[None, dict] = None,
            fed_strategy_params: Union[None, dict] = None,
            workflow_params: Union[None, dict] = None,
            fit_mode: str = 'fed', save_dir_path: str = './fedimp/'
    ):
        # check if fit mode is supported
        if fit_mode not in ['local', 'central', 'fed']:
            raise ValueError(f"Fit mode {fit_mode} not supported")

        # check if imputer and fed strategy are supported and set the imputer and fed strategy names
        if imputer in ['fed_ice', 'fed_mean', 'fed_em']:
            imputer_name = imputer.split('_')[1]
            fed_strategy_name = 'fedavg'
            workflow_name = imputer_name
        elif imputer in ['fed_missforest']:
            imputer_name = imputer.split('_')[1]
            fed_strategy_name = 'fedtree'
            workflow_name = 'ice'
        elif imputer in ['miwae', 'gain']:
            imputer_name = imputer
            workflow_name = 'jm'
            if fed_strategy in ['fedavg', 'fedprox', 'fedavg_ft']:
                fed_strategy_name = fed_strategy
            else:
                raise ValueError(f"Federated strategy {fed_strategy} not supported for imputer {imputer}")
        else:
            raise ValueError(f"Imputer {imputer} not supported")

        # reset fed strategy name if fit mode is local or central
        if fit_mode in ['local', 'central']:
            fed_strategy_name = fit_mode

        # add to the configuration
        self.imputer_name = imputer_name
        self.fed_strategy_name = fed_strategy_name
        self.workflow_name = workflow_name

        # set default values
        if fed_strategy_params is None:
            fed_strategy_params = {}
        if imputer_params is None:
            imputer_params = {}
        if workflow_params is None:
            workflow_params = {}
        self.imputer_params = imputer_params
        self.fed_strategy_params = fed_strategy_params
        self.workflow_params = workflow_params

        # save a directory path
        self.env_dir_path = save_dir_path

    def setup(
            self, clients_train_data: List[np.ndarray], clients_test_data: List[np.ndarray],
            clients_train_data_ms: List[np.ndarray], clients_seeds: List[int], data_config: dict, verbose: int = 0
    ):

        # setup clients
        clients_data = list(zip(clients_train_data, clients_test_data, clients_train_data_ms))
        if verbose > 0:
            print(f"Setting up clients...")

        self.clients = setup_clients(
            clients_data, clients_seeds, data_config,
            imp_model_name=self.imputer_name, imp_model_params=self.imputer_params,
            fed_strategy=self.fed_strategy_name, fed_strategy_client_params=self.fed_strategy_params,
            client_config={'local_dir_path': self.env_dir_path}
        )

        # setup server
        if verbose > 0:
            print(f"Setting up server...")
        self.server = setup_server(
            fed_strategy=self.fed_strategy_name, fed_strategy_params=self.fed_strategy_params, server_config={}
        )

        # setup workflow
        if verbose > 0:
            print(f"Setting up workflow...")
        self.workflow = load_workflow(self.workflow_name, self.workflow_params)

        # evaluator, tracker, result analyzer
        self.evaluator = Evaluator({})  # initialize evaluator
        self.tracker = Tracker()  # initialize tracker
        self.result_analyzer = ResultAnalyzer()  # initialize result analyzer

        if verbose > 0:
            print(f"Environment setup complete.")

    def setup_from_simulator(self, simulator: Simulator, verbose: int = 0):

        self.setup(
            simulator.clients_train_data, simulator.clients_test_data, simulator.clients_train_data_ms,
            simulator.clients_seeds, simulator.data_config, verbose
        )

    def run_fed_imputation(self, run_type: str = 'sequential'):

        ###########################################################################################################
        # Run Federated Imputation
        self.workflow.run_fed_imp(self.clients, self.server, self.evaluator, self.tracker, run_type)

    def save_env(self):
        pass

    def load_env(self):
        pass

    def reset_env(self):
        pass
