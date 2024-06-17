import sys
from typing import Tuple, List, Union, Dict

import loguru
import numpy as np
from .data_partition import load_data_partition
from .missing_simulate import add_missing
from ..utils.reproduce_utils import setup_clients_seed


class Simulator:

    def __init__(self, debug_mode: bool = False):

        # data
        self.data = None
        self.data_config = None

        # clients data
        self.clients_train_data = None
        self.clients_test_data = None
        self.clients_train_data_ms = None
        self.global_test = None
        self.client_seeds = None
        self.stats = None

        # parameters
        self.debug_mode = debug_mode

    def simulate_scenario(
            self,
            data: np.array,
            data_config: dict,
            num_clients: int,
            dp_strategy: str = 'iid-even',
            dp_split_cols: Union[str, int, List[int]] = 'target',
            dp_niid_alpha: float = 0.1,
            dp_size_niid_alpha: float = 0.1,
            dp_min_samples: Union[float, int] = 50,
            dp_max_samples: Union[float, int] = 2000,
            dp_even_sample_size: int = 1000,
            dp_sample_iid_direct: bool = False,
            dp_local_test_size: float = 0.1,
            dp_global_test_size: float = 0.1,
            dp_local_backup_size: float = 0.05,
            dp_reg_bins: int = 50,
            ms_mech_type: str = 'mcar',
            ms_cols: Union[str, List[int]] = 'all',
            obs_cols: Union[str, List[int]] = 'random',
            ms_global_mechanism: bool = False,
            ms_mr_dist_clients: str = 'randu-int',
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
        :param dp_split_cols:
        :param data: data: numpy ndarray
        :param data_config: data configuration dictionary
        :param num_clients: number of clients
        :param dp_strategy:
        :param dp_split_cols:
        :param dp_niid_alpha:
        :param dp_size_niid_alpha:
        :param dp_min_samples:
        :param dp_max_samples:
        :param dp_even_sample_size:
        :param dp_sample_iid_direct:
        :param dp_local_test_size:
        :param dp_global_test_size:
        :param dp_local_backup_size:
        :param dp_reg_bins:
        :param ms_mech_type:
        :param ms_global_mechanism:
        :param ms_mr_dist_clients:
        :param ms_mf_dist_clients:
        :param ms_mm_dist_clients:
        :param ms_missing_features:
        :param ms_mr_lower:
        :param ms_mr_upper:
        :param ms_mm_funcs_bank:
        :param ms_mm_strictness:
        :param ms_mm_obs:
        :param ms_mm_feature_option:
        :param ms_mm_beta_option:
        :param ms_cols: columns indices for adding missing values, support list of column indices or
            string options - 'all' (add to all columns), 'all-num' (add to all numerical columns)
        :param obs_cols: fully observed columns indices for MAR missing mechanism,
            support list of column indices or string options - 'random', 'rest'

        :param data_partition_params: data partition parameters
            - partition_strategy: partition strategy - iid, niid_dir
            - size_strategy: size strategy - even, even2, dir, hs
            - size_niid_alpha: size niid alpha
            - min_samples: minimum samples
            - max_samples: maximum samples
            - niid_alpha: non-iid alpha dirichlet
            - even_sample_size: even sample size
            - sample_iid_direct: sample iid data directly - default: False
            - local_test_size: local test ratio - default: 0.1
            - global_test_size: global test ratio - default: 0.1
            - local_backup_size: local backup_size -  default: 0.1
            - reg_bins: regression bins
        :param missing_simulate_params: missing data simulation parameters
            - global_missing: whether simulate missing data globally or locally
            - mf_strategy: missing features strategy - all
            - mr_dist: missing ratio distribution - fixed, uniform, uniform_int, gaussian, gaussian_int
            - mr_lower: missing ratio lower bound
            - mr_upper: missing ratio upper bound
            - mm_funcs_dist: missing mechanism functions distribution - identity, random, random2,
            - mm_funcs_bank: missing mechanism functions banks - None, 'lr', 'mt', 'all'
            - mm_mech: missing mechanism - 'mcar', 'mar_quantile', 'mar_sigmoid', 'mnar_quantile', 'mnar_sigmoid'
            - mm_strictness: missing adding probailistic or deterministic
            - mm_obs:  missing adding based on observed data
            - mm_feature_option: missing mechanism associated with which features - self, all, allk=0.1
            - mm_beta_option: mechanism beta coefficient option - (mnar) self, sphere, randu, (mar) fixed, randu, randn
        :param seed:
        :return:
        """
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
        # ========================================================================================
        # data partition
        clients_train_data_list, clients_backup_data_list, clients_test_data_list, global_test_data, stats = (
            load_data_partition(
                data, data_config, num_clients,
                split_cols_option=dp_split_cols,
                partition_strategy=dp_strategy,
                seeds=client_seeds,
                niid_alpha=dp_niid_alpha,
                size_niid_alpha=dp_size_niid_alpha,
                min_samples=dp_min_samples, max_samples=dp_max_samples,
                even_sample_size=dp_even_sample_size,
                sample_iid_direct=dp_sample_iid_direct,
                local_test_size=dp_local_test_size, global_test_size=dp_global_test_size,
                local_backup_size=dp_local_backup_size,
                reg_bins=dp_reg_bins, global_seed=global_seed,
            )
        )

        # ========================================================================================
        # simulate missing data
        if isinstance(ms_cols, str):
            if ms_cols == 'all':
                ms_cols = list(range(data.shape[1] - 1))
            elif ms_cols == 'all-num':
                assert 'num_cols' in data_config, 'num_cols not found in data_config'
                ms_cols = data_config['num_cols']
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

        if verbose > 0:
            print("Missing data simulation...")
        client_train_data_ms_list = add_missing(
            clients_train_data_list, ms_cols, client_rngs,
            obs_cols=obs_cols,
            global_missing=ms_global_mechanism,
            mf_strategy=ms_missing_features, mf_dist=ms_mf_dist_clients,
            mr_dist=ms_mr_dist_clients, mr_lower=ms_mr_lower, mr_upper=ms_mr_upper,
            mm_funcs_dist=ms_mm_dist_clients, mm_funcs_bank=ms_mm_funcs_bank, mm_mech=ms_mech_type,
            mm_strictness=ms_mm_strictness, mm_obs=ms_mm_obs, mm_feature_option=ms_mm_feature_option,
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
        }

    def simulate_scenario_lite(
            self, data: np.array, data_config: dict, num_clients: int,
            dp_strategy: str = 'iid-even',
            ms_scenario: str = 'mcar',
            dp_split_col_option: str = 'target',
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
        :param data: data: numpy ndarray
        :param data_config: data configuration dictionary
        :param num_clients: number of clients
        :param dp_strategy: data partition strategy - iid-even, iid-dir@<alpha>, niid-dir@<alpha>, niid-path@<k>
        :param ms_scenario: predefined missing data scenario - mcar, mar-heter, mar-homo,mnar-heter,mnar-homo
        :param dp_split_col_option: iid/niid column strategy partition base on - 'target', 'feature'
        :param ms_cols: missing columns - 'all', 'all-num', 'random'
        :param obs_cols: fully obsevered columns for MAR - 'rest', 'random'
        :param dp_min_samples: minimum sample sizes for clients
        :param dp_max_samples: maximum sample sizes for clients
        :param ms_mr_lower:  minimum missing ratio for each feature
        :param ms_mr_upper: maxinum missing ratio for each feature
        :param verbose: whether verbose the simulation process
        :param seed: random seed
        :return:
        """

        ##################################################################################################
        # Partition Strategy - iid-even, iid-dir@0.1, niid-dir@0.1, niid-path@2
        dp_strategy, dp_params = dp_strategy.split('@')
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
            # assert isinstance(dp_params, int), "Invalid data partition strategy."
            # dp_max_samples = dp_params

        ################################################################################################
        # Data Partition - Split Columns Option
        if dp_split_col_option == 'target':
            dp_split_cols = data.shape[1] - 1
        elif dp_split_col_option == 'feature':
            dp_split_cols = 0
        else:
            raise ValueError(f"Invalid data partition split columns option.")

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
            ms_mr_dist_clients = 'randu-int'

        elif ms_scenario == 'mar-homo':
            ms_mech_type = 'mar_sigmoid'
            ms_global_mechanism = True
            ms_mr_dist_clients = 'randu-int'
            ms_mm_beta_option = 'fixed'
            ms_mm_obs = True

        elif ms_scenario == 'mar-heter':
            ms_mech_type = 'mar_sigmoid'
            ms_global_mechanism = False
            ms_mr_dist_clients = 'randu-int'
            ms_mm_dist_clients = 'random2'
            ms_mm_beta_option = 'randu'
            ms_mm_obs = True

        elif ms_scenario == 'mnar-homo':
            ms_mech_type = 'mnar_sigmoid'
            ms_global_mechanism = True
            ms_mr_dist_clients = 'randu-int'
            ms_mm_beta_option = 'self'

        elif ms_scenario == 'mnar-heter':
            ms_mech_type = 'mnar_sigmoid'
            ms_global_mechanism = False
            ms_mr_dist_clients = 'randu-int'
            ms_mm_beta_option = 'self'
            ms_mm_dist_clients = 'random2'

        elif ms_scenario == 'mnar2-homo':
            ms_mech_type = 'mar_sigmoid'
            ms_global_mechanism = True
            ms_mr_dist_clients = 'randu-int'
            ms_mm_beta_option = 'randu'
            ms_mm_obs = False

        elif ms_scenario == 'mnar2-heter':
            ms_mech_type = 'mar_sigmoid'
            ms_global_mechanism = False
            ms_mr_dist_clients = 'randu-int'
            ms_mm_beta_option = 'randu'
            ms_mm_obs = False
            ms_mm_dist_clients = 'random2'

        return self.simulate_scenario(
            data, data_config, num_clients,
            dp_strategy=dp_strategy,
            dp_split_cols=dp_split_cols,
            dp_niid_alpha=dp_niid_alpha,
            dp_size_niid_alpha=dp_size_niid_alpha,
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

    def save(self, save_path: str):
        pass

    def load(self, load_path: str):
        pass

    def export_data(self):
        pass

    def summarize(self):
        pass

    def visualization(self):
        pass
