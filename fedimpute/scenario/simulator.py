import sys
from typing import Tuple, List, Union, Dict, Any

import loguru
import numpy as np
import pandas as pd
import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

from .data_partition import load_data_partition
from .missing_simulate import add_missing
from ..utils.reproduce_utils import setup_clients_seed
from fedimpute.utils.format_utils import dataframe_to_numpy
from fedimpute.scenario.utils import DistanceComputation
from fedimpute.scenario.missing_simulate.utils import MECH_NAME_MAPPING


class Simulator:

    """
    Simulator class for simulating or constructing missing data scenarios in federated learning environment

    Attributes:
        data (np.ndarray): data to be used for simulation
        data_config (dict): data configuration dictionary
        clients_train_data (List[np.ndarray]): list of clients training data
        clients_test_data (List[np.ndarray]): list of clients test data
        clients_train_data_ms (List[np.ndarray]): list of clients training data with missing values
        global_test (np.ndarray): global test data
        client_seeds (List[int]): list of seeds for clients
        stats (dict): simulation statistics
        debug_mode (bool): whether to enable debug mode
    """

    def __init__(self, debug_mode: bool = False):

        # data
        self.data = None
        self.data_config = None

        # clients data
        self.clients_train_data: List[np.ndarray] = None
        self.clients_test_data: List[np.ndarray] = None
        self.clients_train_data_ms: List[np.ndarray] = None
        self.global_test: np.ndarray = None
        self.clients_seeds: List[int] = None
        self.stats: dict = None

        # parameters
        self.debug_mode = debug_mode
        if debug_mode:
            loguru.logger.remove()
            loguru.logger.add(sys.stdout, level="DEBUG")
        else:
            loguru.logger.remove()
            loguru.logger.add(sys.stdout, level="INFO")
        
        ################################################################################################
        # configuration parameters
        ################################################################################################
        # data partition
        self.dp_strategy: str = 'iid-even'  
        self.dp_split_cols: Union[str, int, List[int]] = 'target'
        self.dp_niid_alpha: float = 0.1
        self.dp_size_niid_alpha: float = 0.1
        self.dp_min_samples: Union[float, int] = 50
        self.dp_max_samples: Union[float, int] = 2000
        self.dp_even_sample_size: int = 1000
        self.dp_sample_iid_direct: bool = False
        self.dp_local_test_size: float = 0.1
        self.dp_global_test_size: float = 0.1
        self.dp_local_backup_size: float = 0.05
        self.dp_reg_bins: int = 50
        
        # missing data
        self.ms_mech_type: str = 'mcar'
        self.ms_cols: Union[str, List[int]] = 'all'
        self.obs_cols: Union[str, List[int]] = 'random'
        self.ms_global_mechanism: bool = False
        self.ms_mr_dist_clients: str = 'randu-int'
        self.ms_mf_dist_clients: str = 'identity'  # TODO
        self.ms_mm_dist_clients: str = 'random'
        self.ms_missing_features: str = 'all'
        self.ms_mr_lower: float = 0.3
        self.ms_mr_upper: float = 0.7
        self.ms_mm_funcs_bank: str = 'lr'
        self.ms_mm_strictness: bool = True
        self.ms_mm_obs: bool = False
        self.ms_mm_feature_option: str = 'allk=0.2'
        self.ms_mm_beta_option: str = None
        
        # seed
        self.seed: int = 100330201
        self.verbose: int = 0

    def create_simulated_scenario(
        self,
        data: Union[np.array, pd.DataFrame],
        data_config: dict,
        num_clients: int,
        dp_strategy: str = 'iid-even',
        dp_split_cols: Union[str, int] = 'target',
        dp_min_samples: Union[float, int] = 50,
        dp_max_samples: Union[float, int] = 2000,
        dp_sample_iid_direct: bool = False,
        dp_local_test_size: float = 0.1,
        dp_global_test_size: float = 0.1,
        dp_local_backup_size: float = 0.05,
        dp_reg_bins: int = 50,
        ms_mech_type: str = 'mcar',
        ms_cols: Union[str, List[int]] = 'all',
        obs_cols: Union[str, List[int]] = 'random',
        ms_global_mechanism: bool = False,
        ms_mr_dist_clients: str = 'randu',
        ms_mf_dist_clients: str = 'identity',  # TODO
        ms_mm_dist_clients: str = 'random',
        ms_missing_features: str = 'all',
        ms_mr_lower: float = 0.3,
        ms_mr_upper: float = 0.7,
        ms_mm_funcs_bank: str = 'lr',
        ms_mm_strictness: bool = True,
        ms_mm_obs: bool = False,
        ms_mm_feature_option: str = 'allk=0.2',
        ms_mm_beta_option: str = None,
        seed: int = 100330201,
        verbose: int = 0
    ) -> Dict[str, List[np.ndarray]]:

        """
        Simulate missing data scenario

        Args:
            data (Union[np.array, pd.DataFrame]): data to be used for simulation
            data_config (dict): data configuration dictionary
            num_clients (int): number of clients
            dp_strategy (str): data partition strategy, default: 'iid-even'
                    - `iid-even`, `iid-dir`, `niid-dir`, `niid-path`
            dp_split_cols (Union[str, int, List[int]]): split columns option
                    - `target`, `feature`, default: `target`
            dp_min_samples (Union[float, int]): minimum samples for clients, default: 50
            dp_max_samples (Union[float, int]): maximum samples for clients, default: 2000
            dp_sample_iid_direct (bool): sample iid data directly, default: False
            dp_local_test_size (float): local test size ratio, default: 0.1
            dp_global_test_size (float): global test size ratio, default: 0.1
            dp_local_backup_size (float): local backup size ratio, default: 0.05
            dp_reg_bins (int): regression bins, default: 50
            ms_mech_type (str): missing mechanism type, default: 'mcar'
                    - `mcar`, `mar_sigmoid`, `mnar_sigmoid`, `mar_quantile`, `mnar_quantile`
            ms_cols (Union[str, List[int]]): missing columns, default: 'all' - `all`, `all-num`, `random`
            obs_cols (Union[str, List[int]]): fully observed columns for MAR, default: 'random' - `random`, `rest`
            ms_global_mechanism (bool): global missing mechanism, default: False
            ms_mr_dist_clients (str): missing ratio distribution, default: 'randu-int' - 'fixed', 'uniform', 'uniform_int', 'gaussian', 'gaussian_int'
            ms_mf_dist_clients (str): missing features distribution, default: 'identity' - 'identity', 'random', 'random2'
            ms_mm_dist_clients (str): missing mechanism functions distribution, default: 'random' - 'identity', 'random', 'random2'
            ms_missing_features (str): missing features strategy, default: 'all' - 'all', 'all-num'
            ms_mr_lower (float): minimum missing ratio for each feature, default: 0.3
            ms_mr_upper (float): maximum missing ratio for each feature, default: 0.7
            ms_mm_funcs_bank (str): missing mechanism functions banks, default: 'lr' - None, 'lr', 'mt', 'all'
            ms_mm_strictness (bool): missing adding probabilistic or deterministic, default: True
            ms_mm_obs (bool): missing adding based on observed data, default: False
            ms_mm_feature_option (str): missing mechanism associated with which features, default: 'allk=0.2' - 'self', 'all', 'allk=0.1'
            ms_mm_beta_option (str): mechanism beta coefficient option, default: None - (mnar) self, sphere, randu, (mar) fixed, randu, randn
            seed (int): random seed, default: 100330201
            verbose (int): whether verbose the simulation process, default: 0

        Returns:
            dict: dictionary of clients training data, test data, training data with missing values, global test data
        """

        ################################################################################################
        # set parameters
        ################################################################################################
        self.dp_strategy = dp_strategy
        self.dp_split_cols = dp_split_cols
        self.dp_min_samples = dp_min_samples
        self.dp_max_samples = dp_max_samples
        self.dp_sample_iid_direct = dp_sample_iid_direct
        self.dp_local_test_size = dp_local_test_size
        self.dp_global_test_size = dp_global_test_size
        self.dp_local_backup_size = dp_local_backup_size
        self.dp_reg_bins = dp_reg_bins
        
        self.ms_mech_type = ms_mech_type
        self.ms_cols = ms_cols
        self.obs_cols = obs_cols
        self.ms_global_mechanism = ms_global_mechanism
        self.ms_mr_dist_clients = ms_mr_dist_clients
        self.ms_mf_dist_clients = ms_mf_dist_clients
        self.ms_mm_dist_clients = ms_mm_dist_clients
        self.ms_missing_features = ms_missing_features
        self.ms_mr_lower = ms_mr_lower
        self.ms_mr_upper = ms_mr_upper
        self.ms_mm_funcs_bank = ms_mm_funcs_bank
        self.ms_mm_strictness = ms_mm_strictness
        self.ms_mm_obs = ms_mm_obs
        self.ms_mm_feature_option = ms_mm_feature_option
        self.ms_mm_beta_option = ms_mm_beta_option
        
        self.seed = seed
        self.verbose = verbose

        ################################################################################################
        # Main Simulation
        ################################################################################################
        if isinstance(data, pd.DataFrame):
            data = dataframe_to_numpy(data, data_config)

        if self.debug_mode:
            loguru.logger.remove()
            loguru.logger.add(sys.stdout, level="DEBUG")
        else:
            loguru.logger.remove()
            loguru.logger.add(sys.stdout, level="INFO")

        # ========================================================================================
        # setup clients seeds
        global_seed = seed
        global_rng = np.random.default_rng(seed)
        client_seeds = setup_clients_seed(num_clients, rng=global_rng)
        client_rngs = [np.random.default_rng(seed) for seed in client_seeds]

        if verbose > 0:
            print("Data partitioning...")
        ################################################################################################
        # Data Partition
        ##################################################################################################
        # Partition Strategy - iid-even, iid-dir@0.1, niid-dir@0.1, niid-path@2
        if '@' in dp_strategy:
            dp_strategy, dp_params = dp_strategy.split('@')
        else:
            dp_strategy = dp_strategy
            dp_params = ''
        
        if dp_strategy not in ['iid-even', 'iid-dir', 'niid-dir', 'niid-path']:
            raise ValueError(f"Invalid data partition strategy.")

        if dp_params != '':
            try:
                dp_params = float(dp_params)
            except ValueError:
                raise ValueError(f"Invalid data partition strategy.")

        dp_size_niid_alpha, dp_niid_alpha = 0.1, 0.1
        if dp_strategy == 'iid-dir':
            assert isinstance(dp_params, float), "Invalid data partition strategy."
            dp_size_niid_alpha = dp_params
        elif dp_strategy == 'niid-dir':
            assert isinstance(dp_params, float), "Invalid data partition strategy."
            dp_niid_alpha = dp_params
        elif dp_strategy == 'niid-path':
            raise NotImplementedError("Not implemented yet.")
        elif dp_strategy == 'iid-even':
            pass
        else:
            raise ValueError(f"Invalid data partition strategy.")
        
        ###############################################################################################
        # DP split cols
        ################################################################################################
        # Data Partition - Split Columns Option
        if isinstance(dp_split_cols, str):
            if dp_split_cols == 'target':
                dp_split_cols = data.shape[1] - 1
            elif dp_split_cols == 'feature':  # TODO: make it to be the most correlated feature among targets
                dp_split_cols = 0
            else:
                raise ValueError(f"Invalid data partition split columns option.")
        elif isinstance(dp_split_cols, int):
            dp_split_cols = dp_split_cols
        # elif isinstance(dp_split_cols, list):  TODO: need to support this
        #     assert all(
        #         isinstance(col, int) for col in dp_split_cols
        #     ), "Invalid data partition split columns option."
        #     dp_split_cols = dp_split_cols
        else:
            raise ValueError(f"Invalid data partition split columns option.")
            
        # ========================================================================================
        # data partition
        clients_train_data_list, clients_backup_data_list, clients_test_data_list, global_test_data, stats = (
            load_data_partition(
                data, 
                data_config, 
                num_clients,
                split_cols_option=dp_split_cols,
                partition_strategy=dp_strategy,
                seeds=client_seeds,
                niid_alpha=dp_niid_alpha,
                size_niid_alpha=dp_size_niid_alpha,
                min_samples=dp_min_samples, 
                max_samples=dp_max_samples,
                sample_iid_direct=dp_sample_iid_direct,
                local_test_size=dp_local_test_size, 
                global_test_size=dp_global_test_size,
                local_backup_size=dp_local_backup_size,
                reg_bins=dp_reg_bins, 
                global_seed=global_seed,
            )
        )

        # ========================================================================================
        # simulate missing data
        if 'num_cols' not in data_config:
            data_config['num_cols'] = data.shape[1] - 1
            
        if isinstance(ms_cols, str):
            if ms_cols == 'all':
                ms_cols = list(range(data.shape[1] - 1))
            elif ms_cols == 'all-num':
                assert 'num_cols' in data_config, 'num_cols not found in data_config'
                ms_cols = list(range(data_config['num_cols']))
            else:
                raise ValueError(f'Invalid ms_cols options: {ms_cols}')
        else:
            list(ms_cols).sort()
            if max(ms_cols) >= data.shape[1] - 1 or min(ms_cols) < 0:
                raise ValueError(f'Invalid indices in "ms_cols" out of data dimension: {ms_cols}')

        if isinstance(obs_cols, str):
            if obs_cols == 'rest':
                obs_cols = list(set(range(data.shape[1] - 1)) - set(ms_cols))
                if len(obs_cols) == 0:
                    obs_cols = [ms_cols[-1]]
                obs_cols.sort()
            elif obs_cols == 'random':
                np.random.seed(seed)
                obs_cols = np.random.choice(range(data.shape[1] - 1), size=1, replace=False)
                obs_cols.sort()
        elif isinstance(obs_cols, list):
            obs_cols.sort()
            if max(obs_cols) >= data.shape[1] - 1 or min(obs_cols) < 0:
                raise ValueError(f'Invalid indices in "obs_cols" out of data dimension: {obs_cols}')
        else:
            raise ValueError(f'Invalid obs_cols options, it should be list of indices or options ("random", "rest")')

        print("Missing data simulation...")
        
        client_train_data_ms_list = add_missing(
            clients_train_data_list, 
            ms_cols, 
            client_rngs,
            obs_cols=obs_cols,
            global_missing=ms_global_mechanism,
            mf_strategy=ms_missing_features, 
            mf_dist=ms_mf_dist_clients,
            mr_dist=ms_mr_dist_clients, 
            mr_lower=ms_mr_lower, 
            mr_upper=ms_mr_upper,
            mm_funcs_dist=ms_mm_dist_clients, 
            mm_funcs_bank=ms_mm_funcs_bank, 
            mm_mech=ms_mech_type,
            mm_strictness=ms_mm_strictness, 
            mm_obs=ms_mm_obs, 
            mm_feature_option=ms_mm_feature_option,
            mm_beta_option=ms_mm_beta_option, seed=global_seed
        )

        # ========================================================================================
        # organize results
        clients_train_data, clients_test_data, clients_train_data_ms = [], [], []
        for i in range(num_clients):
            # merge backup data
            client_train_data = np.concatenate([clients_train_data_list[i], clients_backup_data_list[i]], axis=0)
            client_train_data_ms = np.concatenate(
                [client_train_data_ms_list[i], clients_backup_data_list[i][:, :-1]], axis=0
            )
            client_test_data = clients_test_data_list[i]

            # append data back to a list
            clients_train_data.append(client_train_data)
            clients_test_data.append(client_test_data)
            clients_train_data_ms.append(client_train_data_ms)

        self.stats = stats
        self.clients_train_data = clients_train_data
        self.clients_test_data = clients_test_data
        self.clients_train_data_ms = clients_train_data_ms
        self.global_test = global_test_data
        self.clients_seeds = client_seeds
        self.data = data
        self.data_config = data_config

        if verbose > 0:
            print("Simulation done. Using summary function to check the simulation results.")

        return {
            'clients_train_data': clients_train_data,
            'clients_test_data': clients_test_data,
            'clients_train_data_ms': clients_train_data_ms,
            'clients_seeds': client_seeds,
            'global_test_data': global_test_data,
            'data_config': data_config,
            'stats': stats
        }

    def create_simulated_scenario_lite(
        self, 
        data: Union[np.array, pd.DataFrame], 
        data_config: Dict, 
        num_clients: int,
        dp_strategy: str = 'iid-even',
        ms_scenario: str = 'mcar',
        dp_split_cols: Union[str, int] = 'target',
        ms_cols: Union[str, List[int]] = 'all',
        obs_cols: Union[str, List[int]] = 'random',
        dp_min_samples: Union[float, int] = 50,
        dp_max_samples: Union[float, int] = 8000,
        ms_mr_lower: float = 0.3,
        ms_mr_upper: float = 0.7,
        seed: int = 100330201,
        verbose: int = 0,
    ):
        """
        Simulate missing data scenario

        Args:
            data (np.array): data to be used for simulation
            data_config (dict): data configuration dictionary
            num_clients (int): number of clients
            dp_strategy (str): data partition strategy, default: 'iid-even' - `iid-even`, `iid-dir`, `niid-dir`, `niid-path`
            ms_scenario (str): predefined missing data scenario, default: 'mcar'
                                - `mcar`, `mar-heter`， `mar-homo`, `mnar-heter`, `mnar-homo`
            dp_split_col_option (str): iid/niid column strategy partition base on - 'target', 'feature', default: 'target'
            ms_cols (Union[str, List[int]]): missing columns, default: 'all' - 'all', 'all-num', 'random'
            obs_cols (Union[str, List[int]]): fully observed columns for MAR, default: 'random' - 'random', 'rest'
            dp_min_samples (Union[float, int]): minimum samples for clients, default: 50
            dp_max_samples (Union[float, int]): maximum samples for clients, default: 8000
            ms_mr_lower (float): minimum missing ratio for each feature, default: 0.3
            ms_mr_upper (float): maximum missing ratio for each feature, default: 0.7
            seed (int): random seed, default: 100330201
            verbose (int): whether verbose the simulation process, default: 0

        Returns:
            dict: dictionary of clients training data, test data, training data with missing values, global test data
        """

        ################################################################################################
        # Predefined Missing Scenario - mcar, mar, mnar
        ms_mech_type = 'mcar'
        ms_global_mechanism = False
        ms_mr_dist_clients = 'randu-int'
        ms_mm_dist_clients = 'identity'
        ms_mm_beta_option = None
        ms_mm_obs = False

        if ms_scenario == 'mcar':
            ms_mech_type = 'mcar'
            ms_global_mechanism = False
            ms_mr_dist_clients = 'randu'

        elif ms_scenario == 'mar-homo':
            ms_mech_type = 'mar_logit'
            ms_global_mechanism = True
            ms_mr_dist_clients = 'randu'
            ms_mm_beta_option = 'fixed'
            ms_mm_obs = True

        elif ms_scenario == 'mar-heter':
            ms_mech_type = 'mar_logit'
            ms_global_mechanism = False
            ms_mr_dist_clients = 'randu'
            ms_mm_dist_clients = 'random'
            ms_mm_beta_option = 'randu'
            ms_mm_obs = True

        elif ms_scenario == 'mnar-homo':
            ms_mech_type = 'mnar_sm_logit'
            ms_global_mechanism = True
            ms_mr_dist_clients = 'randu'
            ms_mm_beta_option = 'self'

        elif ms_scenario == 'mnar-heter':
            ms_mech_type = 'mnar_sm_logit'
            ms_global_mechanism = False
            ms_mr_dist_clients = 'randu'
            ms_mm_beta_option = 'self'
            ms_mm_dist_clients = 'random'

        elif ms_scenario == 'mnar2-homo':
            ms_mech_type = 'mar_logit'
            ms_global_mechanism = True
            ms_mr_dist_clients = 'randu'
            ms_mm_beta_option = 'randu'
            ms_mm_obs = False

        elif ms_scenario == 'mnar2-heter':
            ms_mech_type = 'mar_logit'
            ms_global_mechanism = False
            ms_mr_dist_clients = 'randu'
            ms_mm_beta_option = 'randu'
            ms_mm_obs = False
            ms_mm_dist_clients = 'random'

        return self.create_simulated_scenario(
            data, 
            data_config, 
            num_clients,
            dp_strategy=dp_strategy,
            dp_split_cols=dp_split_cols,
            dp_min_samples=dp_min_samples,
            dp_max_samples=dp_max_samples,
            ms_mech_type=ms_mech_type,
            ms_cols=ms_cols,
            obs_cols=obs_cols,
            ms_global_mechanism=ms_global_mechanism,
            ms_mr_dist_clients=ms_mr_dist_clients,
            ms_mm_dist_clients=ms_mm_dist_clients,
            ms_mr_lower=ms_mr_lower,
            ms_mr_upper=ms_mr_upper,
            ms_mm_obs=ms_mm_obs,
            ms_mm_beta_option=ms_mm_beta_option,
            seed=seed,
            verbose=verbose
        )
        
    def create_real_scenario(
        self, 
        datas: List[pd.DataFrame],
        data_config: Dict,
        seed: int = 100330201,
        verbose: int = 0,
    ):
        
        # clients seed
        global_seed = seed
        global_rng = np.random.default_rng(seed)
        num_clients = len(datas)
        client_seeds = setup_clients_seed(num_clients, rng=global_rng)
        
        # convert data to numpy array
        data_list = []
        for data in datas:
            if isinstance(data, pd.DataFrame):
                data = dataframe_to_numpy(data, data_config).astype(np.float32)
            elif isinstance(data, np.ndarray):
                data = data.astype(np.float32)
            else:
                raise ValueError(f"Invalid data type: {type(data)}")
            
            data_list.append(data)
        
        # load data partition - natural partition
        clients_train_data_list, _, clients_test_data_list, global_test_data, stats = (
            load_data_partition(data_list, data_config, num_clients, client_seeds, local_backup_size=0)
        )
        
        # missing data statistics
        clients_train_data = clients_train_data_list
        clients_test_data = clients_test_data_list
        global_test_data = global_test_data
        clients_train_data_ms = [item.copy() for item in clients_train_data_list]
        
        # save results
        self.stats = stats
        self.clients_train_data = clients_train_data
        self.clients_test_data = clients_test_data
        self.clients_train_data_ms = clients_train_data_ms
        self.global_test = global_test_data
        self.clients_seeds = client_seeds
        self.data = data
        self.data_config = data_config

        if verbose > 0:
            print("Simulation done. Using summary function to check the simulation results.")

        return {
            'clients_train_data': clients_train_data,
            'clients_test_data': clients_test_data,
            'clients_train_data_ms': clients_train_data_ms,
            'clients_seeds': client_seeds,
            'global_test_data': global_test_data,
            'data_config': data_config,
            'stats': stats
        }
        

    def save(self, save_path: str):
        pass

    def load(self, load_path: str):
        pass

    def export_data(self):
        pass
    
    def summarize_scenario(
        self, 
        log_to_file: bool = False, 
        file_path: str = None
    ):
        
        clients_train_data = self.clients_train_data
        clients_test_data = self.clients_test_data
        clients_train_data_ms = self.clients_train_data_ms
        global_test_data = self.global_test
        clients_seeds = self.clients_seeds
        
        ################################################################################################
        # Summarizing Report
        ################################################################################################
        # Calculate total number of clients
        total_clients = len(clients_train_data)
        
        # Create header
        summary = "=" * 66 + "\n"
        summary += "Scenario Summary\n"
        summary += "=" * 66 + "\n"
        
        # Basic statistics
        summary += f"Total clients: {total_clients}\n"
        summary += f"Global Test Data: {global_test_data.shape}\n"
        summary += f"Missing Mechanism Category: {MECH_NAME_MAPPING[self.ms_mech_type]}\n"
        summary += "Clients Data Summary:\n"
        
        # Create table headers
        headers = ["", "Train", "Test", "Miss", "MS Ratio", "MS Feature", "Seed"]
        table_data = []
        
        # Populate table data
        for i in range(total_clients):
            train_shape = clients_train_data[i].shape
            test_shape = clients_test_data[i].shape
            missing_shape = clients_train_data_ms[i].shape
            
            # Calculate miss ratio (assuming it's stored in stats or can be calculated)
            miss_ratio = (
                np.isnan(clients_train_data_ms[i]).sum() / (clients_train_data_ms[i].shape[0] * clients_train_data_ms[i].shape[1])
            )
            
            # Calculate missing features (assuming it's stored in stats or can be calculated)
            N_cols = clients_train_data_ms[i].shape[1]
            mask = np.isnan(clients_train_data_ms[i])
            mask_cols = mask.sum(axis=0)
            cols_missing = (mask_cols > 0).sum()
            missing_features = f"{cols_missing}/{N_cols}"
            
            row = [
                f"C{i+1}",
                f"({train_shape[0]},{train_shape[1]})",
                f"({test_shape[0]},{test_shape[1]})",
                f"({missing_shape[0]},{missing_shape[1]})",
                f"{miss_ratio:.2f}",
                f"{missing_features}",
                f"{clients_seeds[i]}"
            ]
            table_data.append(row)
        
        # Format table using tabulate
        table = tabulate.tabulate(
            table_data,
            headers=headers,
            #tablefmt="github",
            stralign="center",
            numalign="center",
            floatfmt=".2f"
        )
        
        summary += table + "\n"
        summary += "=" * 66 + "\n"
        # summary += "*Train, Test, Miss: (n, p)\n"
        # summary += "*MS Ratio: Total Missing Ratio\n"
        # summary += "*MS Feature: #Missing Features/#Total Features\n"
        
        if log_to_file:
            with open(file_path, 'w') as file:
                file.write(summary)
        else:
            print(summary)

    def show_missing_data_details(
        self, 
        client_id: int,
        ms_ratio: bool = False,
        ms_feature: bool = False,
    ):
        pass

    def visualize_missing_pattern(
        self, 
        client_ids: List[int],
        dpi: int = 300,
        fontsize: int = 18
    ):
        n_rows = ceil(len(client_ids) / 4)
        n_cols = 4
        # Create subplot grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), dpi=dpi)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Create custom colormap for binary values
        cmap = plt.cm.get_cmap('rocket')
        colors = [cmap(0), cmap(1.0)]  # Light blue for 0, darker blue for 1
        custom_cmap = plt.matplotlib.colors.ListedColormap(colors)
        
        # Plot each client's missing pattern
        for idx, client_id in enumerate(client_ids):
            row = idx // 4
            col = idx % 4
            
            # Get missing pattern for current client
            mask = np.isnan(self.clients_train_data_ms[client_id])
            pattern = pd.DataFrame(mask).astype(int).astype(str).apply(lambda x: ''.join(x), axis=1)
            sorted_mask = pd.DataFrame(mask).reindex(pattern.sort_values().index).values
            
            # Create heatmap
            sns.heatmap(
                sorted_mask, 
                ax=axes[row, col],
                #cmap=custom_cmap,
                cbar=False,  # Remove colorbar
                vmin=0,
                vmax=1,
                #xticklabels=True,
                #yticklabels=True
            )
            
            axes[row, col].set_title(f'Client {client_id}', fontsize=fontsize, pad=10, fontweight='bold')
            axes[row, col].set_xlabel('Features', fontsize=fontsize, fontweight='bold')
            
            # Only show x-axis labels for bottom row
            if row != n_rows-1:
                axes[row, col].set_xticklabels([])
        
        # Remove empty subplots if any
        for idx in range(len(client_ids), n_rows * 4):
            row = idx // 4
            col = idx % 4
            axes[row, col].remove()
        
        # Add custom legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=colors[0], edgecolor='black', label='Observed'),
            plt.Rectangle((0,0),1,1, facecolor=colors[1], edgecolor='black', label='Missing')
        ]
        fig.legend(
            handles=legend_elements, 
            loc='center',
            bbox_to_anchor=(0.5, -0.1),  # Position at bottom center
            ncol=2,  # Place legend items horizontally
            frameon=False,
            prop={'size': fontsize, 'weight': 'bold'},  # Use prop dictionary for font properties
        )
        
        #plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust layout to make room for legend
        plt.show()
        
    def visualize_missing_distribution(
        self, 
        client_ids: List[int],
        feature_ids: List[int],
        dpi: int = 600,
        fontsize: int = 18,
        bins: int = 30,
        stat: str = 'density',
        kde: bool = False
    ):
        
        client_train_datas = [self.clients_train_data[client_id] for client_id in client_ids]
        client_train_data_ms = [self.clients_train_data_ms[client_id] for client_id in client_ids]
        masks = [np.isnan(client_train_data_ms[i]) for i in range(len(client_ids))]
        
        n_features = len(feature_ids)
        n_clients = len(client_ids)
        n_cols = 4
        n_rows = ceil(n_features / n_cols)*n_clients
        
        #colors = ['#2196F3', '#FF5722']  # Material
        colors = ['#4e79a7', '#f28e2b']  # Contrasting soft
        #colors = ['#2C699A', '#BA2C2C']  # Professional
        #colors = ['lightgrey', 'black']
        colors = ['tab:orange', 'tab:blue']  # Material

        missing_color = colors[0]
        observed_color = colors[1]

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), dpi=dpi)
        for i in range(n_clients):
            for j in range(n_features):
                idx = i*n_features + j
                row = idx // n_cols
                col = idx % n_cols
                
                # missing value distribution
                mask_i = masks[i][:, j]
                missing_values = client_train_datas[i][mask_i, j]
                observed_values = client_train_datas[i][~mask_i, j]
                
                data = pd.DataFrame({
                    'values': np.concatenate([missing_values, observed_values]),
                    'type': np.concatenate([np.zeros(len(missing_values)), np.ones(len(observed_values))])
                })
                
                data['type'] = data['type'].map({1: 'Missing', 0: 'Observed'})
                
                sns.histplot(
                    data, ax=axes[row, col], x='values', hue='type', legend=False, kde=kde,
                    stat = stat, bins = bins, 
                    palette = {
                        'Missing': missing_color,
                        'Observed': observed_color
                    },
                    line_kws = {
                        'linewidth': 2
                    },
                    alpha = 0.6
                )
                axes[row, col].set_title(f'Feature {j+1}', fontsize=fontsize, fontweight='bold')
                if col == 0:
                    axes[row, col].set_ylabel(f'Client {i}', fontsize=fontsize, fontweight='bold')
                else:
                    axes[row, col].set_ylabel('')
                
                axes[row, col].set_xlabel('')
        
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=observed_color, edgecolor='black', label='Observed'),
            plt.Rectangle((0,0),1,1, facecolor=missing_color, edgecolor='black', label='Missing')
        ]
        fig.legend(
            handles=legend_elements, 
            loc='center',
            bbox_to_anchor=(0.5, 0.05),  # Position at bottom center
            ncol=2,  # Place legend items horizontally
            frameon=False,
            prop={'size': fontsize, 'weight': 'bold'},  # Use prop dictionary for font properties
        )
            
            
        plt.show()

    def visualize_data_heterogeneity(
        self, 
        client_ids: List[int],
        distance_method: str = 'swd',
        pca_col_threshold: int = 20,
        dpi: int = 300,
        fontsize: int = 18,
        title: bool = True
    ):
        DISTANCE_METHOD_NAME = {
            'swd': 'Sliced Wasserstein Distance',
            'correlation': 'Correlation Distance',
        }
        
        clients_data = [self.clients_train_data[client_id] for client_id in client_ids]
        
        distance_matrix = DistanceComputation.compute_distance_matrix(
            clients_data, 
            distance_method, 
            pca_col_threshold
        )
        
        n_clients = len(client_ids)
        fig, ax = plt.subplots(figsize=(1*n_clients, 0.9*n_clients), dpi=dpi)
        ax = DistanceComputation.show_distance_matrix(distance_matrix, ax, fontsize)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize = fontsize-3, fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize = fontsize-3, fontweight='bold')
        if title:
            ax.set_title(f'{DISTANCE_METHOD_NAME[distance_method]}', fontsize=fontsize, fontweight='bold')
        plt.show()
        