import os

import numpy as np

from ..base import BaseMLImputer
from collections import OrderedDict
import pickle


class SimpleImputer(BaseMLImputer):

    def __init__(
            self,
            strategy: str = 'mean'
    ):
        super().__init__()
        if strategy not in ['mean']:
            raise ValueError(f"Strategy {strategy} not supported")
        self.strategy: str = strategy
        self.mean_params: np.array = None
        self.model_type = 'numpy'
        self.model_persistable = True
        self.name = 'simple'

    def get_imp_model_params(self, params: dict) -> OrderedDict:
        """
        Return model parameters
        :param params: dict contains parameters for get_imp_model_params
        :return: OrderedDict - model parameters dictionary
        """
        return OrderedDict({"mean": self.mean_params})

    def set_imp_model_params(self, updated_model_dict: OrderedDict, params: dict) -> None:
        """
        Set model parameters
        :param updated_model_dict: global model parameters dictionary
        :param params: parameters for set parameters function
        :return: None
        """
        self.mean_params = updated_model_dict['mean']

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
        self.mean_params = np.zeros(data_utils['n_features'])

    def fit(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> dict:
        """
        Fit imputer to train local imputation models
        :param X: features - numpy array float type
        :param y: target
        :param missing_mask: missing mask
        :param params: parameters for local training
        :return: fit results of local training
        """
        X_ms = X.copy()
        X_ms[missing_mask] = np.nan
        if self.strategy == 'mean':
            self.mean_params = np.nanmean(X_ms, axis=0)
        else:
            raise ValueError(f"Strategy {self.strategy} not supported")

        return {'loss': 0, 'sample_size': X.shape[0]}

    def impute(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> np.ndarray:
        """
        Impute missing values using an imputation model
        :param X: numpy array of features
        :param y: numpy array of target
        :param missing_mask: missing mask
        :param params: parameters for imputation
        :return: imputed data - numpy array - same dimension as X
        """
        # Iterate through all columns
        for i in range(X.shape[1]):
            # Get the mask for current column
            column_mask = missing_mask[:, i]
            # Replace missing values with the mean of the column
            X[column_mask, i] = self.mean_params[i]

        return X
