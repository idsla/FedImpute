import os
import pickle
from collections import OrderedDict
from sklearn.model_selection import StratifiedKFold
from ..base.ice_imputer import ICEImputerMixin
from ..base.base_imputer import BaseMLImputer
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from ..model_loader_utils import load_sklearn_model
import joblib


class MissForestImputer(BaseMLImputer, ICEImputerMixin):

    def __init__(
            self,
            n_estimators:int = 200,
            bootstrap: bool = True,
            n_jobs: int = 2,
            clip: bool = True,
            use_y: bool = False,
    ):
        super().__init__()

        # estimator for numerical and categorical columns
        # TODO: add more parameters for random forest model
        self.clip = clip
        self.min_values = None
        self.max_values = None
        self.use_y = use_y
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs

        # Imputation models
        self.imp_models = None
        self.mm_model = None
        self.data_utils_info = None
        self.seed = None
        self.model_type = 'sklearn'
        self.model_persistable = False
        self.name = 'missforest'

    def initialize(
            self, X: np.array, missing_mask: np.array, data_utils: dict, params: dict, seed: int
    ) -> None:
        """
        Initialize imputer - statistics imputation models etc.
        :param X: data with initially imputed values
        :param missing_mask: missing mask of data
        :param data_utils:  utils dictionary - contains information about data
        :param params: params for initialization
        :param seed: int - seed for randomization
        :return: None
        """

        # initialized imputation models
        self.imp_models = []
        for i in range(data_utils['n_features']):
            if i < data_utils['num_cols']:
                estimator = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    bootstrap=self.bootstrap,
                    n_jobs=self.n_jobs,
                    random_state=seed
                )
            else:
                estimator = RandomForestClassifier(
                    n_estimators=self.n_estimators,
                    bootstrap=self.bootstrap,
                    n_jobs=self.n_jobs, class_weight='balanced', random_state=seed
                )

            X_train = X[:, np.arange(X.shape[1]) != i][0:10]
            y_train = X[:, i][0:10]
            estimator.fit(X_train, y_train)

            self.imp_models.append(estimator)

        # initialize min max values for a clipping threshold
        self.min_values, self.max_values = self.get_clip_thresholds(data_utils)
        self.seed = seed
        self.data_utils_info = data_utils

    def set_imp_model_params(self, updated_model_dict: OrderedDict, params: dict) -> None:
        """
        Set model parameters
        :param updated_model_dict: global model parameters dictionary
        :param params: parameters for set parameters function
            - feature idx
        :return: None
        """
        if 'feature_idx' not in params:
            raise ValueError("Feature index not found in params")
        feature_idx = params['feature_idx']
        imp_model = self.imp_models[feature_idx]
        imp_model.estimators_ = updated_model_dict['estimators']

    def get_imp_model_params(self, params: dict) -> OrderedDict:
        """
        Return model parameters
        :param params: dict contains parameters for get_imp_model_params
            - feature_idx
        :return: OrderedDict - model parameters dictionary
        """
        if 'feature_idx' not in params:
            raise ValueError("Feature index not found in params")
        feature_idx = params['feature_idx']
        imp_model = self.imp_models[feature_idx]
        if 'estimators_' not in imp_model.__dict__:
            return OrderedDict({"estimators": []})
        else:
            return OrderedDict({"estimators": imp_model.estimators_})

    def fit(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> dict:
        """
        Fit imputer to train local imputation models
        :param X: features - float numpy array
        :param y: target
        :param missing_mask: missing mask
        :param params: parameters for local training
            - feature_idx
        :return: fit results of local training
        """
        try:
            feature_idx = params['feature_idx']
        except KeyError:
            raise ValueError("Feature index not found in params")

        row_mask = missing_mask[:, feature_idx]

        X_train = X[~row_mask][:, np.arange(X.shape[1]) != feature_idx]
        y_train = X[~row_mask][:, feature_idx]

        # fit linear imputation models
        estimator = self.imp_models[feature_idx]
        estimator.fit(X_train, y_train)

        return {
            'loss': {},
            'sample_size': X_train.shape[0]
        }

    def impute(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> np.ndarray:
        """
        Impute missing values using an imputation model
        :param X: numpy array of features
        :param y: numpy array of target
        :param missing_mask: missing mask
        :param params: parameters for imputation
        :return: imputed data - numpy array - same dimension as X
        """

        if 'feature_idx' not in params:
            raise ValueError("Feature index not found in params")
        feature_idx = params['feature_idx']

        if self.clip:
            min_values = self.min_values
            max_values = self.max_values
        else:
            min_values = np.full((X.shape[1],), 0)
            max_values = np.full((X.shape[1],), 1)

        row_mask = missing_mask[:, feature_idx]
        if np.sum(row_mask) == 0:
            return X

        # impute missing values
        X_test = X[row_mask][:, np.arange(X.shape[1]) != feature_idx]
        estimator = self.imp_models[feature_idx]
        imputed_values = estimator.predict(X_test)
        if feature_idx >= self.data_utils_info['num_cols']:
            imputed_values = (imputed_values >= 0.5).float()
        imputed_values = np.clip(imputed_values, min_values[feature_idx], max_values[feature_idx])
        X[row_mask, feature_idx] = np.squeeze(imputed_values)

        return X

    def save_model(self, model_path: str, version: str) -> None:
        """
        Save the imputer model
        :param version: version key of model
        :param model_path: path to save the model
        :return: None
        """
        pass

    def load_model(self, model_path: str, version: str) -> None:
        """
        Load the imputer model
        :param version: version key of a model
        :param model_path: path to load the model
        :return: None
        """
        pass
