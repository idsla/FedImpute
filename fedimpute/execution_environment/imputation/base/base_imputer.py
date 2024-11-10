import os
import pickle
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Tuple, List, Dict, Any

import numpy as np
import torch


class BaseMLImputer(metaclass=ABCMeta):

    """
    Abstract class for the non-NN based imputer to be used in the federated imputation environment
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_imp_model_params(self, params: dict) -> OrderedDict:
        """
        Return model parameters

        Args:
            params: dict contains parameters for get_imp_model_params

        Returns:
            OrderedDict - model parameters dictionary
        """
        pass

    @abstractmethod
    def set_imp_model_params(self, updated_model_dict: OrderedDict, params: dict) -> None:
        """
        Set model parameters

        Args:
            updated_model_dict: global model parameters dictionary
            params: parameters for set parameters function
        """
        pass

    @abstractmethod
    def initialize(
            self, X: np.array, missing_mask: np.array, data_utils: dict, params: dict, seed: int
    ) -> None:
        """
        Initialize imputer - statistics imputation models etc.

        Args:
            X: data with intial imputed values
            missing_mask: missing mask of data
            data_utils: data utils dictionary - contains information about data
            params: params for initialization
            seed: int - seed for randomization
        """
        pass

    @abstractmethod
    def fit(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> dict:
        """
        Fit imputer to train local imputation models

        Args:
            X: np.array - float numpy array features
            y: np.array - target
            missing_mask: np.array - missing mask
            params: parameters for local training
        """
        pass

    @abstractmethod
    def impute(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> np.ndarray:
        """
        Impute missing values using an imputation model

        Args:
            X (np.array): numpy array of features
            y (np.array): numpy array of target
            missing_mask (np.array): missing mask
            params (dict): parameters for imputation

        Returns:
            np.ndarray: imputed data - numpy array - same dimension as X
        """
        pass

    def save_model(self, model_path: str, version: str) -> None:
        """
        Save the imputer model

        Args:
            version (str): version key of model
            model_path (str): path to save the model
        """
        params = self.get_imp_model_params({})
        with open(os.path.join(model_path, f'imp_model_{version}.pkl'), 'wb') as f:
            pickle.dump(params, f)

    def load_model(self, model_path: str, version: str) -> None:
        """
        Load the imputer model

        Args:
            version (str): version key of a model
            model_path (str): path to load the model
        """
        with open(os.path.join(model_path, f'imp_model_{version}.pkl'), 'rb') as f:
            params = pickle.load(f)

        self.set_imp_model_params(params, {})


class BaseNNImputer(metaclass=ABCMeta):

    """
    Abstract class for the NN based imputer to be used in the federated imputation environment
    """

    def __init__(self):
        pass

    @abstractmethod
    def get_imp_model_params(self, params: dict) -> OrderedDict:
        """
        Return model parameters

        Args:
            params (dict): dict contains parameters for get_imp_model_params

        Returns:
            OrderedDict - model parameters dictionary
        """
        pass

    @abstractmethod
    def set_imp_model_params(self, updated_model_dict: OrderedDict, params: dict) -> None:
        """
        Set model parameters

        Args:
            updated_model_dict (OrderedDict): global model parameters dictionary
            params (dict): parameters for set parameters function
        """
        pass

    @abstractmethod
    def initialize(
            self, X: np.array, missing_mask: np.array, data_utils: dict, params: dict, seed: int
    ) -> None:
        """
        Initialize imputer - statistics imputation models etc.

        Args:
            X (np.array): data with intial imputed values
            missing_mask (np.array): missing mask of data
            data_utils (dict): data utils dictionary - contains information about data
            params (dict): params for initialization
            seed (int): seed for randomization
        """
        pass

    @abstractmethod
    def configure_model(
            self, params: dict, X: np.ndarray, y: np.ndarray, missing_mask: np.ndarray
    ) -> Tuple[torch.nn.Module, torch.utils.data.DataLoader]:
        """
        Fetch model for training

        Args:
            params (dict): parameters for training
            X (np.ndarray): imputed data
            y (np.ndarray): target
            missing_mask (np.ndarray): missing mask

        Returns:
            Tuple[torch.nn.Module, torch.utils.data.DataLoader]: model, train_dataloader
        """
        pass

    @abstractmethod
    def configure_optimizer(
            self, params: dict, model: torch.nn.Module
    ) -> tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler], Dict[str, Any]]:
        """
        Configure optimizer for training

        Args:
            model (torch.nn.Module): model for training
            params (dict): params for optmizer

        Returns:
            tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LRScheduler], Dict[str, Any]]:
                List of optimizers and List of lr_schedulers, optimizer_params
        """
        pass

    @abstractmethod
    def impute(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> np.ndarray:
        """
        Impute missing values using an imputation model

        Args:
            X (np.array): numpy array of features
            y (np.array): numpy array of target
            missing_mask (np.array): missing mask
            params (dict): parameters for imputation

        Returns:
            np.ndarray: imputed data - numpy array - same dimension as X
        """
        pass

    def save_model(self, model_path: str, version: str) -> None:
        """
        Save the imputer model

        Args:
            version (str): version key of model
            model_path (str): path to save the model
        """
        params = self.get_imp_model_params({})
        torch.save(params, os.path.join(model_path, f'imp_model_{version}.pt'))

    def load_model(self, model_path: str, version: str) -> None:
        """
        Load the imputer model

        Args:
            version (str): version key of a model
            model_path (str): path to load the model
        """
        params = torch.load(os.path.join(model_path, f'imp_model_{version}.pt'))
        self.set_imp_model_params(params, {})
