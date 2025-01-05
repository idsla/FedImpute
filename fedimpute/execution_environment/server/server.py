from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Union, List, Tuple

from scipy import stats

from fedimpute.execution_environment.loaders.load_imputer import load_imputer
from fedimpute.execution_environment.utils.tracker import Tracker
from fedimpute.execution_environment.loaders.load_strategy import load_fed_strategy_server
import numpy as np
from fedimpute.execution_environment.fed_strategy.fed_strategy_server.strategy_base import NNStrategyBaseServer


class Server:

    """
    Server class to be used in the federated imputation environment

    Attributes:
        fed_strategy: str - name of the federated strategy
        fed_strategy_params: dict - parameters of the federated strategy
        server_config: dict - configuration of the server
        X_test_global: np.ndarray - global test data
    """

    def __init__(
            self,
            fed_strategy_name: str,
            fed_strategy_params: dict,
            imputer_name: str,
            imputer_params: dict,
            global_test: np.ndarray,
            data_config: dict,
            server_config: Dict[str, Union[str, int, float]],
            seed: int = 21
    ):

        self.server_config = server_config
        
        # global test data
        self.X_test = global_test[:, :-1]
        self.y_test = global_test[:, -1]
        self.X_test_imp = self.X_test.copy()
        self.X_test_mask = np.isnan(self.X_test)
        self.test_missing = np.any(self.X_test_mask)
        self.seed = seed
        self.data_config = data_config
        self.data_utils = self.calculate_data_utils(data_config)
        
        # imputer
        self.imputer = load_imputer(imputer_name, imputer_params)

        # initialize server side strategy
        self.fed_strategy = load_fed_strategy_server(fed_strategy_name, fed_strategy_params)

        # iniitalize imputer
        self.global_imputer = load_imputer(imputer_name, imputer_params)
        self.global_imputer.initialize(self.X_test, np.isnan(self.X_test), self.data_utils, {}, self.seed)

        if isinstance(self.fed_strategy, NNStrategyBaseServer):
            self.fed_strategy.initialization(self.global_imputer.model, {})
    
    def initial_impute(self, imp_values: np.ndarray, col_type: str = 'num') -> None:
        """
        Initial imputation

        Args:
            imp_values (np.ndarray): imputation values
            col_type (str): column type, 'num' or 'cat'
        """
        num_cols = self.data_utils['num_cols']
        if col_type == 'num':
            for i in range(num_cols):
                if self.X_test_mask[:, i].any():
                    self.X_test_imp[:, i][self.X_test_mask[:, i]] = imp_values[i]
        
        elif col_type == 'cat':
            for i in range(num_cols, self.X_test_imp.shape[1]):
                if self.test_missing:
                    self.X_test_imp[:, i][self.X_test_mask[:, i]] = imp_values[i - num_cols]

        # initialize global imputer after local imputation
        self.imputer.initialize(self.X_test_imp, self.X_test_mask, self.data_utils, {}, self.seed)

    def local_imputation(self, params: dict) -> Union[None, np.ndarray]:
        """
        Perform local imputation

        Args:
            params (dict): instructions for imputation - e.g `temp_imp` for temporary imputation

        Returns:
            Union[None, np.ndarray]: imputed data or None
        """
        global_model_params = self.fed_strategy.get_global_model_params()
        if global_model_params is not None:
            self.imputer.set_imp_model_params(global_model_params, params)
            self.X_test_imp = self.imputer.impute(self.X_test_imp, self.y_test, self.X_test_mask, params)
        
        return None

    def calculate_data_utils(self, data_config: dict) -> dict:
        data_utils = {
            'task_type': data_config['task_type'],
            'n_features': self.X_test.shape[1],
            'num_cols': data_config['num_cols'] if 'num_cols' in data_config else self.X_test.shape[1]
        }

        #########################################################################################################
        # column statistics
        col_stats_dict = {}
        for i in range(self.X_test.shape[1]):
            # numerical stats
            if i < data_utils['num_cols']:
                col_stats_dict[i] = {
                    'min': np.nanmin(self.X_test[:, i]),
                    'max': np.nanmax(self.X_test[:, i]),
                    'mean': np.nanmean(self.X_test[:, i]),
                    'std': np.nanstd(self.X_test[:, i]),
                    'median': np.nanmedian(self.X_test[:, i]),
                }
            # categorical stats
            else:
                col_stats_dict[i] = {
                    'num_class': len(np.unique(self.X_test[:, i][~np.isnan(self.X_test[:, i])])),
                    "mode": stats.mode(self.X_test[:, i][~np.isnan(self.X_test[:, i])], keepdims=False)[0],
                    'mean': np.nanmean(self.X_test[:, i]),
                    'min': np.nanmin(self.X_test[:, i]),
                    'max': np.nanmax(self.X_test[:, i]),
                    # TODO: add frequencies
                }

        data_utils['col_stats'] = col_stats_dict

        #########################################################################################################
        # local data and missing data statistics
        data_utils['sample_size'] = self.X_test.shape[0]
        data_utils['missing_rate_cell'] = np.sum(self.X_test_mask) / (self.X_test.shape[0] * self.X_test.shape[1])
        data_utils['missing_rate_rows'] = np.sum(self.X_test_mask, axis=1) / self.X_test.shape[1]
        data_utils['missing_rate_cols'] = np.sum(self.X_test_mask, axis=0) / self.X_test.shape[0]

        missing_stats_cols = {}
        for col_idx in range(self.X_test.shape[1]):
            row_mask = self.X_test_mask[:, col_idx]
            x_obs_mask = self.X_test_mask[~row_mask][:, np.arange(self.X_test_mask.shape[1]) != col_idx]
            missing_stats_cols[col_idx] = {
                'sample_size_obs': x_obs_mask.shape[0],
                'sample_size_obs_pct': x_obs_mask.shape[0] / self.X_test.shape[0],
                'missing_rate_rows': x_obs_mask.any(axis=1).sum() / x_obs_mask.shape[0],
                'missing_rate_cell': x_obs_mask.sum().sum() / (x_obs_mask.shape[0] * x_obs_mask.shape[1]),
                'missing_rate_obs': x_obs_mask.sum() / (x_obs_mask.shape[0] * x_obs_mask.shape[1]),
            }
        data_utils['missing_stats_cols'] = missing_stats_cols

        #########################################################################################################
        # label stats
        if data_utils['task_type'] == 'regression':
            data_utils['label_stats'] = {
                'min': float(np.nanmin(self.y_test)),
                'max': float(np.nanmax(self.y_test)),
                'mean': float(np.nanmean(self.y_test)),
                'std': float(np.nanstd(self.y_test)),
            }
        else:
            data_utils['label_stats'] = {
                'num_class': len(np.unique(self.y_test))
                # TODO: add frequencies
            }

        return data_utils
    
    def profile(self):

        mask_int = self.X_test_mask.astype(int)
        mask_str_rows = [''.join(map(str, row)) for row in mask_int]
        pattern_counter = Counter(mask_str_rows)

        loguru.logger.debug('-' * 120)
        loguru.logger.debug(
            "Global Test | DS: {} | MaskDS: {} | ImputeDS: {} | MR: {:.2f} |".format(
                self.X_test.shape, self.X_test_mask.shape, self.X_test_imp.shape,
                np.isnan(self.X_test).sum().sum() / (self.X_test.shape[0] * self.X_test.shape[1]),
            ))
        
        if self.test_missing:
            ms_ratio_cols = np.isnan(self.X_test).sum(axis=0) / (self.X_test.shape[0])
            loguru.logger.debug(
                "| MR Cols (Test): {} |".format(np.array2string(ms_ratio_cols, precision=2, suppress_small=True))
            )
