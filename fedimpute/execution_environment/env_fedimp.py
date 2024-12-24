from copy import deepcopy
from typing import List, Union, Tuple
import loguru
import numpy as np
import sys
import datetime

from .loaders.load_environment import setup_clients, setup_server
from .loaders.load_workflow import load_workflow
from .utils.evaluator import Evaluator
from .utils.result_analyzer import ResultAnalyzer
from .utils.tracker import Tracker
from fedimpute.scenario import Simulator
import gc
from fedimpute.utils.reproduce_utils import setup_clients_seed

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
        self.simulator = None
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
        if imputer in ['ice', 'mean', 'em']:
            imputer_name = imputer
            if fed_strategy in ['local', 'central', 'simple_avg']:
                fed_strategy_name = fed_strategy
            else:
                raise ValueError(f"Federated strategy {fed_strategy} not supported for imputer {imputer}")
            #fed_strategy_name = 'simple_avg'
            workflow_name = imputer_name
        
        elif imputer in ['missforest']:
            imputer_name = 'missforest'
            if fed_strategy in ['fedtree', 'local', 'central']:
                fed_strategy_name = fed_strategy
            else:
                raise ValueError(f"Federated strategy {fed_strategy} not supported for imputer {imputer}")
            workflow_name = 'ice'
        
        elif imputer in ['miwae', 'gain', 'notmiwae', 'gnr']:
            imputer_name = imputer
            workflow_name = 'jm'
            if fed_strategy in [
                'fedavg', 'fedavg_ft', 'fedprox', 'scaffold', 'fedadam', 'fedadagrad', 'fedyogi'
            ]:
                fed_strategy_name = fed_strategy
            else:
                raise ValueError(f"Federated strategy {fed_strategy} not supported for imputer {imputer}")
        
        else:
            raise ValueError(f"Imputer {imputer} not supported")

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
        clients_seeds = setup_clients_seed(rng, len(clients_train_data))
        
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
            client_config={'local_dir_path': self.env_dir_path}
        )

        # setup server
        if verbose > 0:
            loguru.logger.info(f"Setting up server...")
        
        self.server = setup_server(
            fed_strategy=self.fed_strategy_name, fed_strategy_params=self.fed_strategy_params,
            imputer_name=self.imputer_name, imputer_params=self.imputer_params,
            global_test=global_test, data_config=data_config, server_config={}
        )

        # setup workflow
        if verbose > 0:
            loguru.logger.info(f"Setting up workflow...")
        self.workflow = load_workflow(self.workflow_name, self.workflow_params)

        # evaluator, tracker, result analyzer
        self.evaluator = Evaluator({})  # initialize evaluator
        self.tracker = Tracker()  # initialize tracker
        self.result_analyzer = ResultAnalyzer()  # initialize result analyzer

        if verbose > 0:
            loguru.logger.info(f"Environment setup complete.")

    def setup_from_simulator(
        self, 
        simulator: Simulator, 
        verbose: int = 0
    ):
        rng = np.random.default_rng(self.seed)
        clients_seeds = setup_clients_seed(rng, len(simulator.clients_train_data))
        
        self.setup_from_data(
            simulator.clients_train_data, 
            simulator.clients_test_data, 
            simulator.clients_train_data_ms,
            simulator.global_test, 
            simulator.data_config, 
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

    def run_fed_imputation(self, run_type: str = 'sequential'):

        ###########################################################################################################
        # Run Federated Imputation
        self.workflow.run_fed_imp(self.clients, self.server, self.evaluator, self.tracker, run_type)

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
        self.simulator = None
        self.evaluator = None
        self.tracker = None
        self.result_analyzer = None
        self.benchmark = None
        self.env_dir_path = None
        
    def get_data(
        self, data_type: str, client_ids: Union[List[int], str] = 'all', include_y: bool = False
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]], np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        
        if isinstance(client_ids, str):
            if client_ids == 'all':
                client_ids = list(range(len(self.clients)))
            else:
                raise ValueError(f"Invalid client ids: {client_ids}")
                    
        if data_type == 'train':
            if include_y:
                return (
                    [self.clients[client_id].X_train for client_id in client_ids],
                    [self.clients[client_id].y_train for client_id in client_ids]
                )
            else:
                return [self.clients[client_id].X_train for client_id in client_ids]
        elif data_type == 'test':
            if include_y:
                return (
                    [self.clients[client_id].X_test for client_id in client_ids],
                    [self.clients[client_id].y_test for client_id in client_ids]
                )
            else:
                return [self.clients[client_id].X_test for client_id in client_ids]
        elif data_type == 'train_imp':
            if include_y:
                raise ValueError("Not y for train_imp data, please set include_y to False")
            return [self.clients[client_id].X_train_imp for client_id in client_ids]
        elif data_type == 'train_mask':
            if include_y:
                raise ValueError("Not y for train_mask data, please set include_y to False")
            return [self.clients[client_id].X_train_mask for client_id in client_ids]
        elif data_type == 'global_test':
            if include_y:
                return (
                    self.server.X_test,
                    self.server.y_test
                )
            else:
                return self.server.X_test
        elif data_type == 'config':
            return self.server.data_config
        else:
            raise ValueError(f"Invalid data type: {data_type}")
