import os
import pickle
from collections import OrderedDict

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from ..base.ice_imputer import ICEImputerMixin
from ..base.base_imputer import BaseMLImputer
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from ..model_loader_utils import load_sklearn_model


class LinearICEImputer(BaseMLImputer, ICEImputerMixin):

    def __init__(
            self,
            estimator_num: str = 'ridge_cv',
            estimator_cat: str = 'logistic',
            mm_model: str = 'logistic',
            mm_model_params=None,
            clip: bool = True,
            use_y: bool = False,
    ):
        super().__init__()

        # model parameters
        self.estimator_num = estimator_num
        self.estimator_cat = estimator_cat
        self.mm_model_name = mm_model
        self.mm_model_params = mm_model_params
        self.clip = clip
        self.min_values = None
        self.max_values = None
        self.use_y = use_y

        # Imputation models
        self.imp_models = None
        self.mm_model = None
        self.data_utils_info = None
        self.seed = None
        self.model_type = 'sklearn'
        self.model_persistable = True
        self.name = 'linear_ice'
        self.model_persistable = False

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

        # initialized imputation models
        self.imp_models = []
        for i in range(data_utils['n_features']):
            if i < data_utils['num_cols']:
                estimator = self.estimator_num
            else:
                estimator = self.estimator_cat

            estimator = load_sklearn_model(estimator)
            X_train = X[:, np.arange(X.shape[1]) != i][0:10]
            y_train = X[:, i][0:10]
            estimator.fit(X_train, y_train)

            self.imp_models.append(estimator)

        # Missing Mechanism Model
        if self.mm_model_name == 'logistic':  # TODO: mm model params
            self.mm_model = LogisticRegressionCV(
                Cs=[1e-1], class_weight='balanced',
                cv=StratifiedKFold(3), random_state=seed, max_iter=1000, n_jobs=-1
            )
        else:
            raise ValueError("Invalid missing mechanism model")

        # initialize min max values for a clipping threshold
        self.min_values, self.max_values = self.get_clip_thresholds(data_utils)

        # seed same as a client
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
        updated_model_dict['w_b'] = np.array(updated_model_dict['w_b'])
        # TODO: make imp model as a class that has get_params() interface so it can using non-sklearn models
        self.imp_models[feature_idx].coef_ = updated_model_dict['w_b'][:-1]
        self.imp_models[feature_idx].intercept_ = updated_model_dict['w_b'][-1]

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
        try:
            parameters = np.concatenate([imp_model.coef_, np.expand_dims(imp_model.intercept_, 0)])
        except AttributeError:
            parameters = np.zeros(self.data_utils_info['n_features'] + 1)
        return OrderedDict({"w_b": parameters})

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
        y_pred = estimator.predict(X_train)
        coef = np.concatenate([estimator.coef_, np.expand_dims(estimator.intercept_, 0)])

        # Fit mechanism models
        # if row_mask.sum() == 0:
        #     mm_coef = np.zeros(X.shape[1]) + 0.001
        # else:
        #     self.mm_model.fit(X, row_mask)
        #     mm_coef = np.concatenate([self.mm_model.coef_[0], self.mm_model.intercept_])

        return {
            'coef': coef,
            'loss': {},
            'sample_size': X_train.shape[0]
        }

    def impute(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> np.ndarray:
        """
        Impute missing values using imputation model
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
        # imp_model_params = []
        # for feature_idx in range(len(self.imp_models)):
        #     params = self.get_imp_model_params({'feature_idx': feature_idx})
        #     imp_model_params.append(params)
        #
        # with open(os.path.join(model_path, f'imp_model_{version}.pkl'), 'wb') as f:
        #     pickle.dump(imp_model_params, f)
        pass

    def load_model(self, model_path: str, version: str) -> None:
        """
        Load the imputer model
        :param version: version key of a model
        :param model_path: path to load the model
        :return: None
        """
        pass
