from copy import deepcopy
from typing import List, Union, Tuple
import loguru
import numpy as np
import sys
import datetime
import pandas as pd

from .loaders.load_environment import setup_clients, setup_server
from .loaders.load_workflow import load_workflow
from .utils.evaluator import Evaluator
from .utils.result_analyzer import ResultAnalyzer
from .utils.tracker import Tracker
from fedimpute.utils.format_utils import arrays_to_dataframes
from fedimpute.scenario import ScenarioBuilder
import gc
from fedimpute.utils.reproduce_utils import setup_clients_seed
from fedimpute.execution_environment.loaders.register import Register

class FedImputeEnv:

    def __init__(
        self,
        debug_mode: bool = False
    ):

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
        self.data_config = {}
        self.seed = None

        # other components
        self.scenario_builder = None
        self.evaluator = None
        self.tracker = None
        self.result_analyzer = None
        self.benchmark = None
        self.env_dir_path = None
        
        # debug mode
        self.debug_mode = debug_mode
        if not self.debug_mode:
            loguru.logger.remove()
            loguru.logger.add(
                sys.stdout,
                format="<level>{message}</level>",
                level="INFO"
            )
        
        # registeration
        self.register = Register()
    

    def configuration(
        self, 
        imputer: str, 
        fed_strategy: Union[str, None] = None,
        imputer_params: Union[None, dict] = None,
        fed_strategy_params: Union[None, dict] = None,
        workflow_params: Union[None, dict] = None,
        seed: int = 100330201,
        save_dir_path: str = './.logs/fedimp/'
    ):

        # check if imputer and fed strategy are supported and set the imputer and fed strategy names
        if imputer not in self.register.get_imputer_mapping():
            raise ValueError(f"Imputer {imputer} not supported")

        imputer_name = imputer
        workflow_name = self.register.get_imputer_workflow_mapping()[imputer]
        
        if imputer in ['miwae', 'gain', 'notmiwae', 'gnr']:
            if fed_strategy in ['local']:
                fed_strategy = 'local_nn'
            elif fed_strategy in ['central']:
                fed_strategy = 'central_nn'

        supported_fed_strategies = self.register.get_imputer_strategy_mapping()[imputer]
        if fed_strategy not in supported_fed_strategies:
            raise ValueError(f"Federated strategy {fed_strategy} not supported for imputer {imputer}")
        fed_strategy_name = fed_strategy

        # add to the configuration
        self.imputer_name = imputer_name
        self.fed_strategy_name = fed_strategy_name
        self.workflow_name = workflow_name
        self.seed = seed

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

    def setup_from_data(
        self, 
        clients_train_data: List[np.ndarray], 
        clients_test_data: List[np.ndarray],
        clients_train_data_ms: List[np.ndarray], 
        global_test: np.ndarray,
        data_config: dict, 
        verbose: int = 0
    ):

        rng = np.random.default_rng(self.seed)
        clients_seeds = setup_clients_seed(len(clients_train_data), rng)
        
        self.data_config = data_config
        if 'num_cols' not in data_config:
            data_config['num_cols'] = clients_train_data[0].shape[1] - 1
        
        # setup clients
        clients_data = list(zip(clients_train_data, clients_test_data, clients_train_data_ms))
        if verbose > 0:
            loguru.logger.info(f"Setting up clients...")

        self.clients = setup_clients(
            clients_data, clients_seeds, data_config,
            imp_model_name=self.imputer_name, imp_model_params=self.imputer_params,
            fed_strategy=self.fed_strategy_name, fed_strategy_client_params=self.fed_strategy_params,
            client_config={'local_dir_path': self.env_dir_path}, register=self.register
        )

        # setup server
        if verbose > 0:
            loguru.logger.info(f"Setting up server...")
        
        self.server = setup_server(
            fed_strategy=self.fed_strategy_name, fed_strategy_params=self.fed_strategy_params,
            imputer_name=self.imputer_name, imputer_params=self.imputer_params,
            global_test=global_test, data_config=data_config, server_config={}, register=self.register
        )

        # setup workflow
        if verbose > 0:
            loguru.logger.info(f"Setting up workflow...")
        self.workflow = self.register.initialize_workflow(self.workflow_name, self.workflow_params)

        # evaluator, tracker, result analyzer
        self.evaluator = Evaluator({})  # initialize evaluator
        self.tracker = Tracker()  # initialize tracker
        self.result_analyzer = ResultAnalyzer()  # initialize result analyzer

        if verbose > 0:
            loguru.logger.info(f"Environment setup complete.")

    def setup_from_scenario_builder(
        self, 
        scenario_builder: ScenarioBuilder, 
        verbose: int = 0
    ):
        rng = np.random.default_rng(self.seed)
        clients_seeds = setup_clients_seed(len(scenario_builder.clients_train_data), rng)
        
        self.setup_from_data(
            scenario_builder.clients_train_data, 
            scenario_builder.clients_test_data, 
            scenario_builder.clients_train_data_ms,
            scenario_builder.global_test, 
            scenario_builder.data_config, 
            verbose
        )

    def clear_env(self):
        del self.clients
        del self.server
        del self.workflow
        del self.evaluator
        del self.tracker
        del self.result_analyzer
        del self.data_config
        
        self.seed = None
        gc.collect()

    def run_fed_imputation(self, run_type: str = 'sequential', verbose: int = 0):

        ###########################################################################################################
        # Run Federated Imputation
        self.workflow.run_fed_imp(self.clients, self.server, self.evaluator, self.tracker, run_type, verbose)

    def show_env_info(self):
        
        if self.clients is None or self.server is None or self.workflow is None:
            raise ValueError("Clients, server, and workflow are not set. Please setup the environment first.")
        
        summary = ""
        summary += "="*60 + "\n"
        summary += f"Environment Information:\n"
        summary += "="*60 + "\n"
        summary += f"Workflow: {self.workflow.name}\n"
        summary += f"Clients:\n"
        for client in self.clients:
            summary += f" - Client {client.client_id}: imputer: {client.imputer.name}, fed-strategy: {client.fed_strategy.name}\n"
        summary += f"Server: fed-strategy: {self.server.fed_strategy.name}\n"
        summary += "="*60 + "\n"
        
        print(summary)
    
    def save_env(self):
        pass

    def load_env(self):
        pass

    def reset_env(self):
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
        self.data_config = {}

        # other components
        self.scenario_builder = None
        self.evaluator = None
        self.tracker = None
        self.result_analyzer = None
        self.benchmark = None
        self.env_dir_path = None
        
    def get_data(
        self, data_type: str, client_ids: Union[List[int], str] = 'all', include_y: bool = False
    ) -> Union[
        List[pd.DataFrame], Tuple[List[pd.DataFrame], List[pd.Series]], pd.DataFrame, Tuple[pd.DataFrame, pd.Series]
    ]:
        
        if isinstance(client_ids, str):
            if client_ids == 'all':
                client_ids = list(range(len(self.clients)))
            else:
                raise ValueError(f"Invalid client ids: {client_ids}")
        columns = self.clients[0].columns
        # original data
        if data_type == 'train':
            if include_y:
                return (
                    [pd.DataFrame(self.clients[client_id].X_train, columns=columns[:-1]) for client_id in client_ids],
                    [pd.Series(self.clients[client_id].y_train, name=columns[-1]) for client_id in client_ids]
                )
            else:
                return [pd.DataFrame(self.clients[client_id].X_train, columns=columns[:-1]) for client_id in client_ids]
        elif data_type == 'test':
            if include_y:
                return (
                    [pd.DataFrame(self.clients[client_id].X_test, columns=columns[:-1]) for client_id in client_ids],
                    [pd.Series(self.clients[client_id].y_test, name=columns[-1]) for client_id in client_ids]
                )
            else:
                return [pd.DataFrame(self.clients[client_id].X_test, columns=columns[:-1]) for client_id in client_ids]
        elif data_type == 'global_test':
            if include_y:
                return (
                    pd.DataFrame(self.server.X_test, columns=columns[:-1]),
                    pd.Series(self.server.y_test, name=columns[-1])
                )
            else:
                return pd.DataFrame(self.server.X_test, columns=columns[:-1])
        # imputed data
        elif data_type == 'train_imp':
            if include_y:
                return (
                    [pd.DataFrame(self.clients[client_id].X_train_imp, columns=columns[:-1]) for client_id in client_ids],
                    [pd.Series(self.clients[client_id].y_train, name=columns[-1]) for client_id in client_ids]
                )
            else:
                return [pd.DataFrame(self.clients[client_id].X_train_imp, columns=columns[:-1]) for client_id in client_ids]
        elif data_type == 'test_imp':
            if include_y:
                return (
                    [pd.DataFrame(self.clients[client_id].X_test_imp, columns=columns[:-1]) for client_id in client_ids],
                    [pd.Series(self.clients[client_id].y_test, name=columns[-1]) for client_id in client_ids]
                )
            else:
                return [pd.DataFrame(self.clients[client_id].X_test_imp, columns=columns[:-1]) for client_id in client_ids]
        elif data_type == 'global_test_imp':
            if include_y:
                return (
                    pd.DataFrame(self.server.X_test_imp, columns=columns[:-1]),
                    pd.Series(self.server.y_test, name=columns[-1])
                )
            else:
                return pd.DataFrame(self.server.X_test_imp, columns=columns[:-1])
        elif data_type == 'train_mask':
            if include_y:
                raise ValueError("Not y for train_mask data, please set include_y to False")
            return [pd.DataFrame(self.clients[client_id].X_train_mask, columns=columns[:-1]) for client_id in client_ids]
        elif data_type == 'global_test_mask':
            if include_y:
                raise ValueError("Not y for global_test_mask data, please set include_y to False")
            return pd.DataFrame(self.server.X_test_mask, columns=columns[:-1])
        elif data_type == 'test_mask':
            if include_y:
                raise ValueError("Not y for test_mask data, please set include_y to False")
            return [pd.DataFrame(self.clients[client_id].X_test_mask, columns=columns[:-1]) for client_id in client_ids]
        # individual data
        elif data_type == 'X_train':
            return [pd.DataFrame(self.clients[client_id].X_train, columns=columns[:-1]) for client_id in client_ids]
        elif data_type == 'y_train':
            return [pd.Series(self.clients[client_id].y_train, name=columns[-1]) for client_id in client_ids]
        elif data_type == 'X_test':
            return [pd.DataFrame(self.clients[client_id].X_test, columns=columns[:-1]) for client_id in client_ids]
        elif data_type == 'y_test':
            return [pd.Series(self.clients[client_id].y_test, name=columns[-1]) for client_id in client_ids]
        elif data_type == 'X_global_test':
            return pd.DataFrame(self.server.X_test, columns=columns[:-1])
        elif data_type == 'y_global_test':
            return pd.Series(self.server.y_test, name=columns[-1])
        elif data_type == 'config':
            return self.server.data_config
        else:
            raise ValueError(f"Invalid data type: {data_type}")
