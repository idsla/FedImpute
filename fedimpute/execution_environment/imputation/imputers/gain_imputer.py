import gc
from collections import OrderedDict
from copy import deepcopy
from typing import Tuple, List, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..base.base_imputer import BaseNNImputer
from ..base.jm_imputer import JMImputerMixin
from ..models.gan_models.gain import GainModel
from ...utils.nn_utils import load_optimizer, load_lr_scheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GAINImputer(BaseNNImputer, JMImputerMixin):

    def __init__(self, imp_model_params: dict, clip: bool = True):
        super().__init__()
        self.name = 'gain'
        self.model_type = 'torch_nn'
        self.imp_model_params = imp_model_params
        self.clip = clip
        self.min_values = None
        self.max_values = None
        self.norm_parameters: Union[dict, None] = None

        # model and solvers
        self.train_dataloader = None
        self.model = None
        self.model_persistable = True

    def initialize(
            self, X: np.array, missing_mask: np.array, data_utils: dict, params: dict, seed: int
    ) -> None:
        """
        Initialize imputer - statistics imputation models etc.
        :param X: data with intial imputed values
        :param missing_mask: missing mask of data
        :param data_utils:  utils dictionary - contains information about data
        :param params: params for initialization
        :param seed: int - seed for randomization
        :return: None
        """

        self.model = GainModel(dim=data_utils['n_features'], **self.imp_model_params)
        self.model.init(seed)

        Xmiss = X.copy()
        Xmiss[missing_mask] = np.nan

        dim = data_utils['n_features']
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        for i in range(dim):
            min_val[i] = np.nanmin(X[:, i])
            max_val[i] = np.nanmax(X[:, i])
        self.norm_parameters = {"min": min_val, "max": max_val}
        self.min_values, self.max_values = self.get_clip_thresholds(data_utils)
        del Xmiss
        gc.collect()

    def get_imp_model_params(self, params: dict) -> OrderedDict:
        return deepcopy(self.model.state_dict())

    def set_imp_model_params(self, updated_model_dict: OrderedDict, params: dict) -> None:
        params_dict = self.model.state_dict()
        params_dict.update(updated_model_dict)
        self.model.load_state_dict(params_dict)

    def configure_model(
            self, params: dict, X: np.ndarray, y: np.ndarray, missing_mask: np.ndarray
    ) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader]:
        """
        Fetch model for training
        :param params: parameters for training
        :param X: imputed data
        :param y: target
        :param missing_mask: missing mask
        :return: model, train_dataloader
        """
        # set up train and test data for a training imputation model
        try:
            batch_size = params['batch_size']
        except KeyError as e:
            raise ValueError(f"Parameter {str(e)} not found in params")

        if self.train_dataloader is not None:
            return self.model, self.train_dataloader
        else:
            n = X.shape[0]
            X_imp = X.copy()
            X_mask = missing_mask.copy()
            bs = min(batch_size, n)

            train_dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(X_imp).float(), torch.from_numpy(~X_mask).float()
            )
            train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
            self.train_dataloader = train_dataloader

            return self.model, train_dataloader

    def configure_optimizer(
            self, params: dict, model: torch.nn.Module
    ) -> tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
        """
        Configure optimizer for training
        :param model: training model
        :param params: params for optmizer
        :return: List of optimizers and List of lr_schedulers
        """
        try:
            optimizer_name = params['optimizer']
            learning_rate = params['learning_rate']
            weight_decay = params['weight_decay']
            scheduler_name = params['scheduler']
            scheduler_params = params['scheduler_params']
        except KeyError as e:
            raise ValueError(f"Parameter {str(e)} not found in params")

        g_solver = load_optimizer(
            optimizer_name, model.generator_layer.parameters(), learning_rate, weight_decay
        )

        d_solver = load_optimizer(
            optimizer_name, model.discriminator_layer.parameters(), learning_rate, weight_decay
        )

        d_lr_scheduler = load_lr_scheduler(scheduler_name, d_solver, scheduler_params)
        g_lr_scheduler = load_lr_scheduler(scheduler_name, g_solver, scheduler_params)

        return (
            [d_solver, g_solver], [d_lr_scheduler, g_lr_scheduler]
        )

    def impute(
            self, X: np.array, y: np.array, missing_mask: np.array, params: dict
    ) -> np.ndarray:

        if self.norm_parameters is None:
            raise RuntimeError("invalid norm_parameters")
        if self.model is None:
            raise RuntimeError("Fit the model first")

        X = torch.from_numpy(X.copy()).float()
        mask = torch.from_numpy(~missing_mask.copy()).float()

        with torch.no_grad():
            self.model.to(DEVICE)
            x_imp = self.model.impute(X, mask)
            self.model.to('cpu')

        if self.clip:
            for i in range(x_imp.shape[1]):
                x_imp[:, i] = np.clip(x_imp[:, i], self.min_values[i], self.max_values[i])

        del X
        gc.collect()

        return x_imp
