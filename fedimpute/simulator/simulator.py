from typing import Tuple, List

import numpy as np
from .data_partition import load_data_partition
from .missing_simulate import add_missing
from ..utils.reproduce_utils import setup_clients_seed


class Simulator:

    def __init__(self, scenario_params: dict):
        self.scenario_params = scenario_params
        self.clients_local_data = None
        self.global_test = None
        self.client_seeds = None
        self.stats = None

    def simulate_scenario(
            self, data: np.array, data_config: dict, num_clients,
            data_partition_params: dict,
            missing_simulate_params: dict,
            seed: int = 100330201,
    ) -> Tuple[List[Tuple[np.array, np.ndarray, np.ndarray]], np.array, List[int], List[List[Tuple[int, int]]]]:
        """
        Simulate missing data scenario
        :param rng: numpy random generator
        :param data: data
        :param data_config: data configuration
        :param num_clients: number of clients
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
        # ========================================================================================
        # setup clients seeds
        global_seed = seed
        global_rng = np.random.default_rng(seed)
        client_seeds = setup_clients_seed(num_clients, rng=global_rng)
        client_rngs = [np.random.default_rng(seed) for seed in client_seeds]

        # ========================================================================================
        # data partition
        clients_train_data_list, clients_backup_data_list, clients_test_data_list, global_test_data, stats = (
            load_data_partition(
                data, data_config, num_clients, seeds=client_seeds, global_seed=global_seed, **data_partition_params
            )
        )

        # ========================================================================================
        # simulate missing data globally
        cols = data_config['ms_col_idx']
        if 'mm_obs' in missing_simulate_params and missing_simulate_params['mm_obs']:
            if 'obs_col_idx' in data_config and len(data_config['obs_col_idx']) > 0:
                obs_cols = np.array(data_config['obs_col_idx'])
                cols = np.array(cols)
                cols = cols[~np.isin(cols, obs_cols)].tolist()
            else:
                # size = (data.shape[1] - 1) // 4
                # cols = global_rng.choice(cols, size=size, replace=False)
                raise ValueError(
                    'obs col is zero or not found, at least one obs col need for mar missing (mm_obs=True)')

        client_train_data_ms_list = add_missing(
            clients_train_data_list, cols, client_rngs, seed=global_seed, **missing_simulate_params
        )

        # ========================================================================================
        # organize results
        clients_data = []
        for i in range(num_clients):
            # merge backup data
            client_train_data = np.concatenate([clients_train_data_list[i], clients_backup_data_list[i]], axis=0)
            client_train_data_ms = np.concatenate(
                [client_train_data_ms_list[i], clients_backup_data_list[i][:, :-1]], axis=0
            )
            client_test_data = clients_test_data_list[i]

            # append data back to a list
            clients_data.append((client_train_data, client_test_data, client_train_data_ms))

        return clients_data, global_test_data, client_seeds, stats

    def save(self):
        pass

    def summarize(self):
        pass

    def visualization(self):
        pass
