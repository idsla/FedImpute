import os.path
from collections import Counter
from typing import Tuple, Union
import numpy as np
from scipy import stats
import torch
import loguru

from ..imputation.base import BaseNNImputer, BaseMLImputer
from ..fed_strategy.fed_strategy_client import NNStrategyBaseClient
from ..loaders.load_imputer import load_imputer
from ..loaders.load_strategy import load_fed_strategy_client
# from ..utils.fed_nn_trainer import fit_fed_nn_model


class Client:

    """
    Client class presenting a client in the federated imputation execution environment, it contains
    the training and testing data, missing data, imputed data, imputation model class, and federated strategy class.

    Attributes:
        client_id (int): client id
        X_train (np.ndarray): training data
        y_train (np.ndarray): training labels
        X_test (np.ndarray): testing data
        y_test (np.ndarray): testing labels
        X_train_ms (np.ndarray): missing data
        X_train_mask (np.ndarray): missing data mask
        X_train_imp (np.ndarray): imputed data
        data_utils (dict): data statistics
        imputer (BaseImputer): imputation model
        fed_strategy (BaseFedStrategy): federated strategy
        seed (int): seed
        client_config (dict): client configuration
    """

    def __init__(
            self,
            client_id: int,
            train_data: np.ndarray,
            test_data: np.ndarray,
            X_train_ms: np.ndarray,
            data_config: dict,
            imp_model_name,
            imp_model_params,
            fed_strategy: str,
            fed_strategy_params: dict,
            client_config: dict,
            seed=0,
    ) -> None:

        # client id
        self.client_id = client_id

        # data
        self.X_train, self.y_train = train_data[:, :-1], train_data[:, -1]  # training data
        self.X_test, self.y_test = test_data[:, :-1], test_data[:, -1]  # testing data
        self.X_train_ms = X_train_ms  # missing data
        
        # imputed training data
        self.X_train_mask = np.isnan(self.X_train_ms)  # missing data mask
        self.X_train_imp = self.X_train_ms.copy()  # imputed data
        self.no_ground_truth = True if np.isnan(self.X_train).any() else False
        
        # imputed testing data
        self.X_test_mask = np.isnan(self.X_test)  # missing data mask
        self.X_test_imp = self.X_test.copy()  # imputed data
        self.test_missing = True if np.sum(self.X_test_mask) > 0 else False

        # calculate data stats
        self.data_utils = self.calculate_data_utils(data_config, data_type='train')
        self.data_utils_test = self.calculate_data_utils(data_config, data_type='test')
        self.profile()

        # imputation model
        self.imputer = load_imputer(imp_model_name, imp_model_params)

        # fed strategy
        self.fed_strategy = load_fed_strategy_client(fed_strategy, fed_strategy_params)

        # others
        self.seed = seed
        self.client_config = client_config
        self.client_local_dir_path = os.path.join(client_config['local_dir_path'], 'client' + str(client_id))
        if not os.path.exists(self.client_local_dir_path):
            os.makedirs(self.client_local_dir_path)

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
                self.X_train_imp[:, i][self.X_train_mask[:, i]] = imp_values[i]
                if self.test_missing:
                    self.X_test_imp[:, i][self.X_test_mask[:, i]] = imp_values[i]
        
        elif col_type == 'cat':
            for i in range(num_cols, self.X_train.shape[1]):
                self.X_train_imp[:, i][self.X_train_mask[:, i]] = imp_values[i - num_cols]
                if self.test_missing:
                    self.X_test_imp[:, i][self.X_test_mask[:, i]] = imp_values[i - num_cols]

        # initialize imputer after local imputation
        self.imputer.initialize(self.X_train_imp, self.X_train_mask, self.data_utils, {}, self.seed)

    def fit_local_imp_model(self, params: dict) -> Tuple[Union[dict, torch.nn.Module], dict]:
        """
        Fit a local imputation model

        Args:
            params (dict): instructions for fitting the imputation model

        Returns:
            Tuple[dict, dict]: model parameters and fitting results dictionary
        """
        if not params['fit_model']:
            if isinstance(self.fed_strategy, NNStrategyBaseClient):
                fit_res = self.fed_strategy.get_fit_res(self.imputer.model, params)
                fit_res.update({'sample_size': self.X_train_imp.shape[0], 'converged': True})
                return self.fed_strategy.get_parameters(self.imputer.model, params), fit_res
            else:
                fit_res = self.imputer.get_fit_res(params)
                fit_res.update({'sample_size': self.X_train_imp.shape[0], 'converged': True})
                return self.imputer.get_imp_model_params(params), fit_res
        else:
            ############################################################################################################
            # NN based Imputation Models
            if isinstance(self.fed_strategy, NNStrategyBaseClient):

                imp_model, fit_res = self.fed_strategy.train_local_nn_model(
                    self.imputer, params, self.X_train_imp, self.y_train, self.X_train_mask
                )

                return imp_model, fit_res
            ############################################################################################################
            # Traditional Imputation Models
            else:
                fit_res = self.imputer.fit(
                    self.X_train_imp, self.y_train, self.X_train_mask, params
                )
                model_parameters = self.imputer.get_imp_model_params(params)
                #fit_res.update(self.data_utils)

            return model_parameters, fit_res

    def update_local_imp_model(self, updated_local_model: Union[dict, None], params: dict) -> None:
        """
        Fit a local imputation model

        Args:
            updated_local_model (Union[dict, None]): updated model parameters
            params (dict): instructions for updating the imputation model
        """
        # if 'update_model' not in params or ('update_model' in params and params['update_model'] == True):
        #     print('update model')
        if updated_local_model is not None:
            if isinstance(self.fed_strategy, NNStrategyBaseClient):
                self.fed_strategy.set_parameters(updated_local_model, self.imputer.model, params)
            else:
                self.imputer.set_imp_model_params(updated_local_model, params)

    def local_imputation(self, params: dict) -> Union[None, np.ndarray]:
        """
        Perform local imputation

        Args:
            params (dict): instructions for imputation - e.g `temp_imp` for temporary imputation

        Returns:
            Union[None, np.ndarray]: imputed data or None
        """
        if 'temp_imp' in params and params['temp_imp']:
            X_train_imp = self.imputer.impute(self.X_train_imp, self.y_train, self.X_train_mask, params)
            return X_train_imp
        else:
            self.X_train_imp = self.imputer.impute(self.X_train_imp, self.y_train, self.X_train_mask, params)
            if self.test_missing:
                self.X_test_imp = self.imputer.impute(self.X_test_imp, self.y_test, self.X_test_mask, params)
            return None

    def save_imp_model(self, version: str) -> None:
        """
        Save imputation model
        """
        # save imp model params
        if self.imputer.model_persistable:
            self.imputer.save_model(self.client_local_dir_path, version)
        # save imp data
        else:
            np.savez_compressed(
                os.path.join(self.client_local_dir_path, f'imp_data_{version}.npz'), imp_data=self.X_train_imp
            )
            
            if self.test_missing:
                np.savez_compressed(
                    os.path.join(self.client_local_dir_path, f'imp_data_test_{version}.npz'), imp_data=self.X_test_imp
                )

    def load_imp_model(self, version: str) -> None:
        """
        Save imputation model
        """
        # load imp model params
        if self.imputer.model_persistable:
            self.imputer.load_model(self.client_local_dir_path, version)
            self.X_train_imp[self.X_train_mask] = 0
            self.X_train_imp = self.imputer.impute(self.X_train_imp, self.y_train, self.X_train_mask, {})
        # load imp data
        else:
            self.X_train_imp = np.load(
                os.path.join(self.client_local_dir_path, f'imp_data_{version}.npz')
            )['imp_data']
            
            if self.test_missing:
                self.X_test_imp = np.load(
                    os.path.join(self.client_local_dir_path, f'imp_data_test_{version}.npz')
                )['imp_data']

    def calculate_data_utils(self, data_config: dict, data_type: str = 'train') -> dict:
        """
        Calculate data statistic
        """
        if data_type == 'train':
            X = self.X_train
            X_mask = self.X_train_mask
            y = self.y_train
        else:
            X = self.X_test
            X_mask = self.X_test_mask
            y = self.y_test

        data_utils = {
            'task_type': data_config['task_type'],
            'n_features': X.shape[1],
            'num_cols': data_config['num_cols'] if 'num_cols' in data_config else X.shape[1],
            'ms_cols_idx': np.where(X_mask.any(axis=0))[0]
        }

        #########################################################################################################
        # column statistics
        col_stats_dict = {}
        for i in range(X.shape[1]):
            # numerical stats
            if i < data_utils['num_cols']:
                col_stats_dict[i] = {
                    'min': np.nanmin(X[:, i]),
                    'max': np.nanmax(X[:, i]),
                    'mean': np.nanmean(X[:, i]),
                    'std': np.nanstd(X[:, i]),
                    'median': np.nanmedian(X[:, i]),
                }
            # categorical stats
            else:
                col_stats_dict[i] = {
                    'num_class': len(np.unique(X[:, i][~np.isnan(X[:, i])])),
                    "mode": stats.mode(X[:, i][~np.isnan(X[:, i])], keepdims=False)[0],
                    'mean': np.nanmean(X[:, i]),
                    'min': np.nanmin(X[:, i]),
                    'max': np.nanmax(X[:, i]),
                    # TODO: add frequencies
                }

        data_utils['col_stats'] = col_stats_dict

        #########################################################################################################
        # local data and missing data statistics
        data_utils['sample_size'] = X.shape[0]
        data_utils['missing_rate_cell'] = np.sum(X_mask) / (X.shape[0] * X.shape[1])
        data_utils['missing_rate_rows'] = np.sum(X_mask, axis=1) / X.shape[1]
        data_utils['missing_rate_cols'] = np.sum(X_mask, axis=0) / X.shape[0]

        missing_stats_cols = {}
        for col_idx in range(X.shape[1]):
            row_mask = X_mask[:, col_idx]
            x_obs_mask = X_mask[~row_mask][:, np.arange(X_mask.shape[1]) != col_idx]
            missing_stats_cols[col_idx] = {
                'sample_size_obs': x_obs_mask.shape[0],
                'sample_size_obs_pct': x_obs_mask.shape[0] / X.shape[0],
                'missing_rate_rows': x_obs_mask.any(axis=1).sum() / x_obs_mask.shape[0],
                'missing_rate_cell': x_obs_mask.sum().sum() / (x_obs_mask.shape[0] * x_obs_mask.shape[1]),
                'missing_rate_obs': x_obs_mask.sum() / (x_obs_mask.shape[0] * x_obs_mask.shape[1]),
            }
        data_utils['missing_stats_cols'] = missing_stats_cols

        #########################################################################################################
        # label stats
        if data_utils['task_type'] == 'regression':
            data_utils['label_stats'] = {
                'min': float(np.nanmin(y)),
                'max': float(np.nanmax(y)),
                'mean': float(np.nanmean(y)),
                'std': float(np.nanstd(y)),
            }
        else:
            data_utils['label_stats'] = {
                'num_class': len(np.unique(y)),
                # TODO: add frequencies
            }

        return data_utils

    def profile(self):

        mask_int = self.X_train_mask.astype(int)
        mask_str_rows = [''.join(map(str, row)) for row in mask_int]
        pattern_counter = Counter(mask_str_rows)

        loguru.logger.debug('-' * 120)
        loguru.logger.debug(
            "| Client {:2} | DS: {} | MissDS: {} | MaskDS: {} | ImputeDS: {} | MR: {:.2f} | MR Test: {:.2f} |".format(
                self.client_id, self.X_train.shape, self.X_train_ms.shape, self.X_train_mask.shape,
                self.X_train_imp.shape,
                np.isnan(self.X_train_ms).sum().sum() / (self.X_train_ms.shape[0] * self.X_train_ms.shape[1]),
                np.isnan(self.X_test).sum().sum() / (self.X_test.shape[0] * self.X_test.shape[1])
            ))

        ms_ratio_cols = np.isnan(self.X_train_ms).sum(axis=0) / (self.X_train_ms.shape[0])
        loguru.logger.debug(
            "| MR Cols: {} |".format(np.array2string(ms_ratio_cols, precision=2, suppress_small=True))
        )
        
        if self.test_missing:
            ms_ratio_cols = np.isnan(self.X_test).sum(axis=0) / (self.X_test.shape[0])
            loguru.logger.debug(
                "| MR Cols (Test): {} |".format(np.array2string(ms_ratio_cols, precision=2, suppress_small=True))
            )
