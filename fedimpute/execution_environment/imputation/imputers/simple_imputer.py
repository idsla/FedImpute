import os

import numpy as np

from ..base import BaseMLImputer
from collections import OrderedDict
import pickle


class SimpleImputer(BaseMLImputer):

    """
    Simple imputer class for imputing missing values in data using simple strategies like mean, median etc.

    Attributes:
        strategy (str): strategy for imputation - mean, median etc.
        mean_params (np.array): mean parameters for imputation
        model_type (str): type of the model - numpy or sklearn
        model_persistable (bool): whether model is persistable or not
        name (str): name of the imputer
    """

    def __init__(
            self,
            strategy: str = 'mean'
    ):
        super().__init__('simple', True)
        if strategy not in ['mean']:
            raise ValueError(f"Strategy {strategy} not supported")
        self.strategy: str = strategy
        self.mean_params: np.array = None
        self.model_type = 'numpy'
        self.model_persistable = True
        self.name = 'simple'
        self.fit_res_history = []

    def get_imp_model_params(self, params: dict) -> OrderedDict:

        return OrderedDict({"mean": self.mean_params})

    def set_imp_model_params(self, updated_model_dict: OrderedDict, params: dict) -> None:

        self.mean_params = updated_model_dict['mean']

    def initialize(
            self, X: np.array, missing_mask: np.array, data_utils: dict, params: dict, seed: int
    ) -> None:

        self.mean_params = np.zeros(data_utils['n_features'])

    def fit(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> dict:

        X_ms = X.copy()
        X_ms[missing_mask] = np.nan
        if self.strategy == 'mean':
            self.mean_params = np.nanmean(X_ms, axis=0)
        else:
            raise ValueError(f"Strategy {self.strategy} not supported")
        
        self.fit_res_history.append({
            'sample_size': X.shape[0],
            'mean_params': self.mean_params,
            'loss': 0
        })

        return {'loss': 0, 'sample_size': X.shape[0]}

    def impute(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> np.ndarray:

        # Iterate through all columns
        for i in range(X.shape[1]):
            # Get the mask for current column
            column_mask = missing_mask[:, i]
            # Replace missing values with the mean of the column
            X[column_mask, i] = self.mean_params[i]

        return X
    
    def get_fit_res(self, params: dict) -> dict:
        return self.fit_res_history[-1]
    
    def __str__(self):
        return f"Simple Imputer"

    def __repr__(self):
        return f"Simple Imputer" 
