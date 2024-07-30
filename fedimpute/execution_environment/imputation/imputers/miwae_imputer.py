import gc
from copy import deepcopy
from typing import Dict, Union, List, Tuple, OrderedDict

import numpy as np
from torch.utils.data import DataLoader

from ..models.vae_models.gnr import GNR
from ..models.vae_models.miwae import MIWAE
from ..models.vae_models.notmiwae import NOTMIWAE
import torch
from ..base import JMImputerMixin, BaseNNImputer

from ...utils.nn_utils import load_optimizer, load_lr_scheduler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MIWAEImputer(BaseNNImputer, JMImputerMixin):

    """
    MiWAE imputer class for imputing missing values in data using Multiple Imputation with Auxiliary Deep Generative Models.

    Attributes:
        name (str): name of the imputer
        clip (bool): whether to clip the imputed values
        latent_size (int): size of the latent space
        n_hidden (int): number of hidden units
        n_hidden_layers (int): number of hidden layers
        out_dist (str): output distribution
        K (int): number of samples
        L (int): number of MCMC samples
        activation (str): activation function
        initializer (str): initializer for weights
        batch_size (int): batch size for training
        learning_rate (int): learning rate for optimizer
        weight_decay (int): weight decay for optimizer
        scheduler (str): scheduler for optimizer
        optimizer (str): optimizer for training
    """

    def __init__(
            self,
            name: str = 'miwae',
            # model params
            latent_size: int = 5,
            n_hidden: int = 16,
            n_hidden_layers: int = 2,
            out_dist='studentt',
            K: int = 20,
            L: int = 100,
            activation='tanh',
            initializer='xavier',
            clip: bool = True,
            # training params
            batch_size: int = 256,
            learning_rate: int = 0.001,
            weight_decay: int = 0.0001,
            scheduler: str = "step",
            optimizer: str = 'adam',
    ):

        super().__init__()
        self.model = None
        self.name = name

        # imputation model parameters
        self.clip = clip
        self.latent_size = latent_size
        self.n_hidden = n_hidden
        self.n_hidden_layers = n_hidden_layers
        self.out_dist = out_dist
        self.K = K
        self.L = L
        self.activation = activation
        self.initializer = initializer

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

        # other parameters
        self.min_values = None
        self.max_values = None
        self.model_type = 'torch_nn'
        self.train_dataloader = None
        self.model_persistable = True
        self.seed = None

    def get_imp_model_params(self, params: dict) -> OrderedDict:

        return deepcopy(self.model.state_dict())

    def set_imp_model_params(self, updated_model_dict: OrderedDict, params: dict) -> None:

        params_dict = self.model.state_dict()
        params_dict.update(deepcopy(updated_model_dict))
        self.model.load_state_dict(params_dict)

    def initialize(
            self, X: np.array, missing_mask: np.array, data_utils: dict, params: dict, seed: int
    ) -> None:

        self.seed = seed
        if self.name == 'miwae':
            self.model = MIWAE(
                num_features=data_utils['n_features'],
                latent_size=self.latent_size,
                n_hidden=self.n_hidden,
                n_hidden_layers=self.n_hidden_layers,
                out_dist=self.out_dist,
                K=self.K,
                L=self.L,
                activation=self.activation,
                initializer=self.initializer,
            )
        # elif self.name == 'notmiwae':
        #     self.model = NOTMIWAE(num_features=data_utils['n_features'], **self.imp_model_params)
        # elif self.model == 'gnr':
        #     self.model = GNR(num_features=data_utils['n_features'], **self.imp_model_params)
        else:
            raise ValueError(f"Model {self.name} not supported")

        self.model.init(seed)
        self.min_values, self.max_values = self.get_clip_thresholds(data_utils)

    def configure_model(
            self, params: dict, X: np.ndarray, y: np.ndarray, missing_mask: np.ndarray
    ) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader]:

        n = X.shape[0]
        X_imp = X.copy()
        X_mask = missing_mask.copy()
        bs = min(self.batch_size, n)

        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_imp).float(), torch.from_numpy(~X_mask).float()
        )
        train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)

        return self.model, train_dataloader

    def configure_optimizer(
            self, params: dict, model: torch.nn.Module
    ) -> tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:

        optimizer = load_optimizer(self.optimizer, model.parameters(), self.learning_rate, self.weight_decay)
        lr_scheduler = load_lr_scheduler(self.scheduler, optimizer, self.scheduler_params)

        return [optimizer], [lr_scheduler]

    def fit(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> dict:

        self.model.to(DEVICE)

        # initialization weights
        # if init:
        # 	self.model.init()

        try:
            lr = params['learning_rate']
            weight_decay = params['weight_decay']
            local_epochs = params['local_epoch']
            batch_size = params['batch_size']
            # verbose = params['verbose']
            optimizer_name = params['optimizer']
        except KeyError as e:
            raise ValueError(f"Parameter {str(e)} not found in params")

        if optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
        else:
            raise NotImplementedError

        # data
        n = X.shape[0]
        X_imp = X.copy()
        X_mask = missing_mask.copy()
        bs = min(batch_size, n)
        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_imp).float(), torch.from_numpy(~X_mask).float()
        )
        train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        # training
        final_loss = 0
        rmses = []
        for ep in range(local_epochs):

            # shuffle data
            # perm = np.random.permutation(n)  # We use the "random reshuffling" version of SGD
            # batches_data = np.array_split(X_imp[perm,], int(n / bs), )
            # batches_mask = np.array_split(X_mask[perm,], int(n / bs), )
            # batches_y = np.array_split(y[perm,], int(n / bs), )
            total_loss, total_iters = 0, 0
            self.model.train()
            # for it in range(len(batches_data)):
            for it, inputs in enumerate(train_dataloader):
                optimizer.zero_grad()
                self.model.encoder.zero_grad()
                self.model.decoder.zero_grad()
                # b_data = torch.from_numpy(batches_data[it]).float().to(DEVICE)
                # b_mask = torch.from_numpy(~batches_mask[it]).float().to(DEVICE)
                b_data, b_mask = inputs
                b_data = b_data.to(DEVICE)
                b_mask = b_mask.to(DEVICE)
                # b_y = torch.from_numpy(batches_y[it]).long().to(DEVICE)
                data = [b_data, b_mask]

                loss, ret_dict = self.model.compute_loss(data)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_iters += 1

            # print loss
            # if (ep + 1) % 1 == 0:
            #     tqdm.write('Epoch %s/%s, Loss = %s' % (
            #     ep, local_epochs, total_loss / total_iters))

            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            final_loss = total_loss / total_iters

            # if (ep + 1) % 10000 == 0:
            #     with torch.no_grad():
            #         X_imp_new = self.model.impute(
            #             torch.from_numpy(X_imp).float().to(DEVICE), torch.from_numpy(~X_mask).float().to(DEVICE)
            #         )
            #         X_imp = X_imp_new.detach().clone().cpu().numpy()

        self.model.to("cpu")

        return {
            'loss': final_loss, 'rmse': rmses, 'sample_size': X.shape[0]
        }

    def impute(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> np.ndarray:

        X_train_imp = X
        X_train_imp[missing_mask] = 0
        x = torch.from_numpy(X_train_imp.copy()).float().to(DEVICE)
        mask = torch.from_numpy(~missing_mask.copy()).float().to(DEVICE)

        self.model.to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            x_imp = self.model.impute(x, mask)

        x_imp = x_imp.detach().cpu().numpy()
        self.model.to("cpu")

        del X
        gc.collect()

        if self.clip:
            for i in range(x_imp.shape[1]):
                x_imp[:, i] = np.clip(x_imp[:, i], self.min_values[i], self.max_values[i])

        return x_imp
