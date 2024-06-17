import os
import pickle
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Tuple, List

import numpy as np
import torch


class BaseMLImputer(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def get_imp_model_params(self, params: dict) -> OrderedDict:
        """
        Return model parameters
        :param params: dict contains parameters for get_imp_model_params
        :return: OrderedDict - model parameters dictionary
        """
        pass

    @abstractmethod
    def set_imp_model_params(self, updated_model_dict: OrderedDict, params: dict) -> None:
        """
        Set model parameters
        :param updated_model_dict: global model parameters dictionary
        :param params: parameters for set parameters function
        :return: None
        """
        pass

    @abstractmethod
    def initialize(
            self, X: np.array, missing_mask: np.array, data_utils: dict, params: dict, seed: int
    ) -> None:
        """
        Initialize imputer - statistics imputation models etc.
        :param X: data with intial imputed values
        :param missing_mask: missing mask of data
        :param data_utils: data utils dictionary - contains information about data
        :param params: params for initialization
        :param seed: int - seed for randomization
        :return: None
        """
        pass

    @abstractmethod
    def fit(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> dict:
        """
        Fit imputer to train local imputation models
        :param X: features - float numpy array
        :param y: target
        :param missing_mask: missing mask
        :param params: parameters for local training
        :return: fit results of local training
        """
        pass

    @abstractmethod
    def impute(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> np.ndarray:
        """
        Impute missing values using an imputation model
        :param X: numpy array of features
        :param y: numpy array of target
        :param missing_mask: missing mask
        :param params: parameters for imputation
        :return: imputed data - numpy array - same dimension as X
        """
        pass

    def save_model(self, model_path: str, version: str) -> None:
        """
        Save the imputer model
        :param version: version key of model
        :param model_path: path to save the model
        :return: None
        """
        params = self.get_imp_model_params({})
        with open(os.path.join(model_path, f'imp_model_{version}.pkl'), 'wb') as f:
            pickle.dump(params, f)

    def load_model(self, model_path: str, version: str) -> None:
        """
        Load the imputer model
        :param version: version key of a model
        :param model_path: path to load the model
        :return: None
        """
        with open(os.path.join(model_path, f'imp_model_{version}.pkl'), 'rb') as f:
            params = pickle.load(f)

        self.set_imp_model_params(params, {})


class BaseNNImputer(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def get_imp_model_params(self, params: dict) -> OrderedDict:
        """
        Return model parameters
        :param params: dict contains parameters for get_imp_model_params
        :return: OrderedDict - model parameters dictionary
        """
        pass

    @abstractmethod
    def set_imp_model_params(self, updated_model_dict: OrderedDict, params: dict) -> None:
        """
        Set model parameters
        :param updated_model_dict: global model parameters dictionary
        :param params: parameters for set parameters function
        :return: None
        """
        pass

    @abstractmethod
    def initialize(
            self, X: np.array, missing_mask: np.array, data_utils: dict, params: dict, seed: int
    ) -> None:
        """
        Initialize imputer - statistics imputation models etc.
        :param X: data with intial imputed values
        :param missing_mask: missing mask of data
        :param data_utils: data utils dictionary - contains information about data
        :param params: params for initialization
        :param seed: int - seed for randomization
        :return: None
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def configure_optimizer(
            self, params: dict, model: torch.nn.Module
    ) -> tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler]]:
        """
        Configure optimizer for training
        :param model: model for training
        :param params: params for optmizer
        :return: List of optimizers and List of lr_schedulers
        """
        pass

    @abstractmethod
    def impute(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> np.ndarray:
        """
        Impute missing values using an imputation model
        :param X: numpy array of features
        :param y: numpy array of target
        :param missing_mask: missing mask
        :param params: parameters for imputation
        :return: imputed data - numpy array - same dimension as X
        """
        pass

    def save_model(self, model_path: str, version: str) -> None:
        """
        Save the imputer model
        :param version: version key of model
        :param model_path: path to save the model
        :return: None
        """
        params = self.get_imp_model_params({})
        torch.save(params, os.path.join(model_path, f'imp_model_{version}.pt'))

    def load_model(self, model_path: str, version: str) -> None:
        """
        Load the imputer model
        :param version: version key of a model
        :param model_path: path to load the model
        :return: None
        """
        params = torch.load(os.path.join(model_path, f'imp_model_{version}.pt'))
        self.set_imp_model_params(params, {})
