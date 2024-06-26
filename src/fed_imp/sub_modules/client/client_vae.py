import torch
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.modules.iterative_imputation.distributed_imputer import DistributedFeatureImputer
from src.modules.evaluation.evaluation import Evaluator
from sklearn.linear_model import LogisticRegression
import numpy as np
from loguru import logger
from src.fed_imp.sub_modules.dataloader import construct_tensor_dataset
import pandas as pd
import matplotlib.pyplot as plt
import missingno
from copy import deepcopy
from typing import Dict, Union, List, Tuple
from torch.utils.data import DataLoader
from ..model.miwae import MIWAE
import torch


class ClientVAE:
    """
    Client class for federated learning
    """

    def __init__(
            self,
            client_id,
            client_data, client_data_config: dict,
            imputation_config: dict,
            debug: bool = False,
            seed: int = 21
    ):
        ################################################################################################################
        # Data
        ################################################################################################################
        self.X_train_filled = None
        self.X_train_filled_pt = None
        self.tol = 1e-3
        self.client_id = client_id

        self.X_train, self.y_train = client_data.get('train_data')[:, :-1], client_data.get('train_data')[:, -1]
        self.X_test, self.y_test = client_data.get('test_data')[:, :-1], client_data.get('test_data')[:, -1]
        self.X_train_ms = client_data.get('train_data_ms')[:, :-1]
        self.client_data_config = client_data_config
        self.seed = seed
        self.convergency = False
        self.features_min = self.X_train.min(axis=0)
        self.features_max = self.X_train.max(axis=0)

        # get original missing mask 0-1 matrix
        self.missing_mask = np.isnan(self.X_train_ms).astype(bool)
        self.num_cols = self.client_data_config.get('num_cols', self.X_train.shape[1])
        self.task_type = self.client_data_config.get('task_type')

        # debug
        self.debug = debug
        # missingno.matrix(pd.DataFrame(self.X_train_ms))
        # plt.show()
        self.top_k_idx = None
        if self.task_type == 'classification':
            self.regression = False
        else:
            self.regression = True

        ################################################################################################################
        # imputation parameters
        ################################################################################################################
        self.initial_strategy_num = imputation_config.get('initial_strategy', 'mean')
        self.imp_clip = imputation_config.get('clip', False)
        imputation_config['imp_params']['clip'] = self.imp_clip
        self.imputation_model = MIWAEImputer(
            imp_model_params=imputation_config['imp_model_params'],
            imp_params=imputation_config['imp_params']
        )

        data_utils = {"n_features": self.X_train_ms.shape[1]}
        self.imputation_model.initialize(data_utils, seed)

        ################################################################################################################
        # evaluation of imputation
        ################################################################################################################
        if self.task_type == 'classification':
            self.imp_evaluation_model = imputation_config.get('imp_evaluation_model', 'logistic')
            self.imp_evaluation_params = imputation_config.get('imp_evaluation_params', {})
            self.evaluation_metrics = ['imp@rmse', 'imp@w2', 'imp@sliced_ws']
        else:
            self.imp_evaluation_model = imputation_config.get('imp_evaluation_model', 'ridge')
            self.evaluation_metrics = ['imp@rmse', 'imp@w2', 'imp@sliced_ws']
            self.imp_evaluation_params = imputation_config.get('imp_evaluation_params', {})

        self.evaluator = Evaluator(
            task_type=self.task_type, metrics=self.evaluation_metrics, model=self.imp_evaluation_model,
            X_train=self.X_train, y_train=self.y_train, X_test=self.X_test, y_test=self.y_test,
            mask=self.missing_mask, seed=seed, tune_params='notune'
        )

    def fit(self, fit_instruction: dict):

        local_epoches = fit_instruction['local_epoches']

        fit_res = (
            self.imputation_model.fit_local_imp_model(
                self.X_train_filled, self.missing_mask, self.y_train, params={
                    'local_epochs': local_epoches
                }
            ))

        model_weights = self.imputation_model.get_imp_model_params()
        losses = fit_res
        missing_info = {"sample_size": self.get_sample_size()}

        return model_weights, losses, missing_info

    def transform(self, transform_task: str, transform_instruction: dict, global_weights):
        if transform_task == 'impute_data':
            self.X_train_filled = self.imputation_model.imputation(
                self.X_train_ms, self.missing_mask, params={},
                features_min=self.features_min, features_max=self.features_max
            )
        elif transform_task == 'update_imp_model':
            update_weights = transform_instruction['update_weights']
            if update_weights:
                self.imputation_model.update_imp_model(global_weights)
        else:
            raise NotImplementedError

    def evaluate(
            self, evaluate_task: str, evaluate_instruction: dict, global_weights=None, update_weights: bool = True
    ):
        if evaluate_task == 'evaluate_imputation_model':
            ret = self.evaluator.evaluation_imp(self.X_train_filled)
            return ret
        elif evaluate_task == 'evaluate_prediction_model':
            return {}
        return {}

    ####################################################################################################################
    # helper functions
    ####################################################################################################################
    def get_initial_values(self):
        ret_mean, ret_max, ret_min, sample_size, ms_info = [], [], [], [], []
        for idx in range(self.X_train_ms.shape[1]):
            ret_mean.append(np.nanmean(self.X_train_ms[:, idx]))
            ret_max.append(np.nanmax(self.X_train_ms[:, idx]))
            ret_min.append(np.nanmin(self.X_train_ms[:, idx]))
            ms_info.append(self.imputation_model.get_missing_info(idx, self.missing_mask)['missing_cell_pct'])
            sample_size.append(self.imputation_model.get_missing_info(idx, self.missing_mask)['sample_row_pct'])
        return np.array(ret_mean), np.array(ret_max), np.array(ret_min), np.array(sample_size), np.array(ms_info)

    def initial_impute(self, aggregated_values):
        if aggregated_values is None:
            self.X_train_filled = self.imputation_model.initial_impute(self.X_train_ms, self.initial_strategy_num)
            self.X_train_filled_pt = self.X_train_filled.copy()
        else:
            self.X_train_filled = self.X_train_ms.copy()
            for idx in range(self.X_train_ms.shape[1]):
                indices = np.where(np.isnan(self.X_train_ms[:, idx]))[0]
                self.X_train_filled[indices, idx] = aggregated_values[idx]

            self.X_train_filled_pt = self.X_train_filled.copy()

    def get_sample_size(self):
        return self.X_train_filled.shape[0]

    def update_imp_model_minmax(self):
        self.imputation_model.features_max = self.features_max
        self.imputation_model.features_min = self.features_min


from tqdm.auto import tqdm, trange
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MIWAEImputer:

    def __init__(self, imp_model_params: dict, imp_params: dict):

        self.model = None
        self.name = "miwae"

        # imputation model parameters
        self.imp_model_params = imp_model_params

        # imputation parameters
        self.lr = imp_params['lr']
        self.weight_decay = imp_params['weight_decay']
        self.batch_size = imp_params['batch_size']
        self.optimizer_name = imp_params['optimizer']
        self.imp_clip = imp_params['clip']

    @staticmethod
    def initial_impute(X, initial_strategy):

       # initialize
        X = X.copy()
        # initial imputation for numerical columns
        if initial_strategy == 'mean':
            simple_imp = SimpleImputer(strategy='mean')
            X_t = simple_imp.fit_transform(X)
        elif initial_strategy == 'median':
            simple_imp = SimpleImputer(strategy='median')
            X_t = simple_imp.fit_transform(X)
        elif initial_strategy == 'zero':
            simple_imp = SimpleImputer(strategy='constant', fill_value=0)
            X_t = simple_imp.fit_transform(X)
        else:
            raise ValueError("initial_strategy_num must be one of 'mean', 'median', 'zero'")

        return X_t

    def get_imp_model_params(self) -> dict:
        """
        Return model parameters
        """
        return deepcopy(self.model.state_dict())

    def update_imp_model(self, updated_model: dict) -> None:
        params = self.model.state_dict()
        params.update(updated_model)
        self.model.load_state_dict(params)

    def initialize(self, data_utils, seed) -> None:
        self.model = MIWAE(num_features=data_utils['n_features'], **self.imp_model_params)
        self.model.init(seed)

    def fit_local_imp_model(
            self, X_train_imp: np.ndarray, X_train_mask: np.ndarray, y_train: np.ndarray, params: dict
    ) -> Dict[str, float]:
        """
        Local training of imputation model for local epochs
        """
        self.model.to(DEVICE)

        # initialization weights
        # if init:
        # 	self.model.init()
        local_epochs = params['local_epochs']

        # optimizer and params
        lr = self.lr
        weight_decay = self.weight_decay
        batch_size = self.batch_size
        optimizer_name = self.optimizer_name

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        else:
            raise NotImplementedError

        # data
        n = X_train_imp.shape[0]
        X_imp = X_train_imp.copy()
        X_mask = X_train_mask.copy()
        bs = min(batch_size, n)

        # training
        final_loss = 0
        rmses = []
        for ep in trange(local_epochs, desc='Client Local Epoch', leave=False, colour='blue'):

            # shuffle data
            perm = np.random.permutation(n)  # We use the "random reshuffling" version of SGD
            batches_data = np.array_split(X_imp[perm,], int(n / bs), )
            batches_mask = np.array_split(X_mask[perm,], int(n / bs), )
            batches_y = np.array_split(y_train[perm,], int(n / bs), )
            total_loss, total_iters = 0, 0
            self.model.train()
            for it in range(len(batches_data)):
                optimizer.zero_grad()
                self.model.encoder.zero_grad()
                self.model.decoder.zero_grad()
                b_data = torch.from_numpy(batches_data[it]).float().to(DEVICE)
                b_mask = torch.from_numpy(~batches_mask[it]).float().to(DEVICE)
                b_y = torch.from_numpy(batches_y[it]).long().to(DEVICE)
                data = [b_data, b_mask]

                loss, ret_dict = self.model.compute_loss(data)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_iters += 1

            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            final_loss = total_loss / total_iters

        self.model.to("cpu")

        return {'loss': final_loss}

    def imputation(
            self, X_train_ms: np.ndarray, X_train_mask: np.ndarray, params: dict, features_min, features_max
    ) -> np.ndarray:
        """
        Impute missing values of client data
        """
        # make complete
        X_train_imp = X_train_ms.copy()
        X_train_imp[X_train_mask] = 0
        self.model.to(DEVICE)
        x = torch.from_numpy(X_train_imp.copy()).float().to(DEVICE)
        mask = torch.from_numpy(~X_train_mask.copy()).float().to(DEVICE)
        with torch.no_grad():
            x_imp = self.model.impute(x, mask)

        x_imp = x_imp.detach().cpu().numpy()
        self.model.to("cpu")

        # clip the values
        if self.imp_clip:
            for i in range(x_imp.shape[1]):
                x_imp[:, i] = np.clip(x_imp[:, i], features_min[i], features_max[i])

        return x_imp

    @staticmethod
    def get_missing_info(col_idx, missing_mask):

        # X train missing mask
        row_mask = missing_mask[:, col_idx]  # row mask
        X_train_mask = missing_mask[~row_mask][:, np.arange(missing_mask.shape[1]) != col_idx]

        sample_size = X_train_mask.shape[0]
        missing_row_pct = X_train_mask.any(axis=1).sum() / X_train_mask.shape[0]
        missing_cell_pct = X_train_mask.sum().sum() / (X_train_mask.shape[0] * X_train_mask.shape[1])
        sample_row_pct = sample_size / missing_mask.shape[0]

        # total pct of missing
        total_missing_cell_pct = missing_mask.sum().sum() / (missing_mask.shape[0] * missing_mask.shape[1])
        total_missing_row_pct = missing_mask.any(axis=1).sum() / missing_mask.shape[0]

        return {
            'sample_size': sample_size,
            'sample_row_pct': sample_row_pct,
            'missing_cell_pct': missing_cell_pct,
            'missing_row_pct': missing_row_pct,
            'total_missing_cell_pct': total_missing_cell_pct,
            'total_missing_row_pct': total_missing_row_pct
        }