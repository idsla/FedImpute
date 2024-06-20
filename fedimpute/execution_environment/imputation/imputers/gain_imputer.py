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

    """
    GAIN imputer class for imputing missing values in data using Generative Adversarial Imputation Networks.

    Attributes:
        h_dim (int): dimension of hidden layers
        n_layers (int): number of layers
        activation (str): activation function
        initializer (str): initializer for weights
        loss_alpha (float): alpha parameter for loss
        hint_rate (float): hint rate for loss
        clip (bool): whether to clip the imputed values
        batch_size (int): batch size for training
        learning_rate (int): learning rate for optimizer
        weight_decay (int): weight decay for optimizer
        scheduler (str): scheduler for optimizer
        optimizer (str): optimizer for training
        scheduler_params (dict): scheduler parameters
    """

    def __init__(
            self,
            h_dim: int = 20,
            n_layers: int = 2,
            activation: str = 'relu',
            initializer: str = 'kaiming',
            loss_alpha: float = 10,
            hint_rate: float = 0.9,
            clip: bool = True,
            # training params
            batch_size: int = 256,
            learning_rate: int = 0.001,
            weight_decay: int = 0.0001,
            scheduler: str = "step",
            optimizer: str = 'sgd',
    ):
        super().__init__()
        self.name = 'gain'
        self.model_type = 'torch_nn'
        self.min_values = None
        self.max_values = None
        self.norm_parameters: Union[dict, None] = None

        # model and solvers
        self.train_dataloader = None
        self.model = None
        self.model_persistable = True

        # model parameters
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.activation = activation
        self.initializer = initializer
        self.loss_alpha = loss_alpha
        self.hint_rate = hint_rate
        self.clip = clip

        # training params
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.scheduler_params = {
            "step_size": 10,
            "gamma": 0.5,
            "schedule_last_epoch": -1
        }
        self.optimizer = optimizer

    def initialize(
            self, X: np.array, missing_mask: np.array, data_utils: dict, params: dict, seed: int
    ) -> None:

        self.model = GainModel(
            dim=data_utils['n_features'],
            h_dim=self.h_dim,
            n_layers=self.n_layers,
            activation=self.activation,
            initializer=self.initializer,
            loss_alpha=self.loss_alpha,
            hint_rate=self.hint_rate
        )
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

        if self.train_dataloader is not None:
            return self.model, self.train_dataloader
        else:
            n = X.shape[0]
            X_imp = X.copy()
            X_mask = missing_mask.copy()
            bs = min(self.batch_size, n)

            train_dataset = torch.utils.data.TensorDataset(
                torch.from_numpy(X_imp).float(), torch.from_numpy(~X_mask).float()
            )
            train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
            self.train_dataloader = train_dataloader

            return self.model, train_dataloader

    def configure_optimizer(
            self, params: dict, model: torch.nn.Module
    ) -> tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:

        g_solver = load_optimizer(
            self.optimizer, model.generator_layer.parameters(), self.learning_rate, self.weight_decay
        )

        d_solver = load_optimizer(
            self.optimizer, model.discriminator_layer.parameters(), self.learning_rate, self.weight_decay
        )

        d_lr_scheduler = load_lr_scheduler(self.scheduler, d_solver, self.scheduler_params)
        g_lr_scheduler = load_lr_scheduler(self.scheduler, g_solver, self.scheduler_params)

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
