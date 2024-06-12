from collections import OrderedDict
from functools import reduce

from sklearn.impute import SimpleImputer

from ..base.ice_imputer import ICEImputerMixin
from ..base.base_imputer import BaseMLImputer
import numpy as np


# TODO:
#  1. singular matrix convergence logic
#  2. global convergence checking logic


class EMImputer(BaseMLImputer, ICEImputerMixin):

    def __init__(
            self,
            clip: bool = True,
            use_y: bool = False
    ):
        super().__init__()

        # estimator for numerical and categorical columns
        self.clip = clip
        self.min_values = None
        self.max_values = None
        self.use_y = use_y
        self.data_utils_info = None
        self.seed = None
        self.model_type = 'simple'
        self.name = 'em'

        # parameters
        self.mu = None
        self.sigma = None
        self.miss = None
        self.obs = None
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

        self.min_values, self.max_values = self.get_clip_thresholds(data_utils)

        # seed same as a client
        self.seed = seed
        self.data_utils_info = data_utils

        # set mu and sigma
        n_row, n_col = X.shape
        one_to_nc = np.arange(1, n_col + 1, step=1)
        self.miss = one_to_nc * missing_mask - 1
        self.obs = one_to_nc * (~missing_mask) - 1
        self.mu = np.nanmean(X, axis=0)
        self.sigma = np.cov(X.T)
        if np.isnan(self.sigma).any():
            self.sigma = np.diag(np.nanvar(X, axis=0))

    def set_imp_model_params(self, updated_model_dict: OrderedDict, params: dict) -> None:

        self.mu = updated_model_dict['mu']
        self.sigma = updated_model_dict['sigma']

    def get_imp_model_params(self, params: dict) -> OrderedDict:

        return OrderedDict({
            'mu': self.mu,
            'sigma': self.sigma
        })

    def fit(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> dict:

        # TODO: SAVE MODEL INTERVAL FOR LOCAL STRATEGY
        local_epochs = params['local_epoch']
        converged = False
        convergence_threshold = params['convergence_thres']

        for iteration in range(local_epochs):
            try:
                mu_new, sigma_new, X_new = self._em(X, self.miss, self.obs, self.mu, self.sigma)

                if self._converged(self.mu, self.sigma, mu_new, sigma_new, convergence_threshold):
                    print(f"EM converged after {iteration} iterations.")
                    converged = True
                    break

                self.mu, self.sigma = mu_new, sigma_new
            except BaseException as e:
                print(f"EM step failed. {e}")
                converged = True
                break

        return {
            'sample_size': X.shape[0],
            'converged': converged
        }

    def impute(self, X: np.array, y: np.array, missing_mask: np.array, params: dict) -> np.ndarray:

        # it has been already imputed in a fit step
        try:
            _, _, X_new = self._em(X, self.miss, self.obs, self.mu, self.sigma)
        except BaseException as e:
            X_new = X.copy()
            print(f"EM step failed. {e}")

        if np.any(np.isnan(X_new)):
            # fallback to mean imputation in case of singular matrix.
            X_new = SimpleImputer(strategy="mean").fit_transform(X_new)

        return X_new

    @staticmethod
    def _em(X, miss, obs, mu, sigma):

        nrows, ncols = X.shape

        mu_tilde, sigma_tilde = {}, {}

        for row_id in range(nrows):
            # update sigma params for calculate X
            sigma_tilde[row_id] = np.zeros((ncols, ncols))

            miss_ids = miss[row_id, :][miss[row_id, :] != -1]
            obs_ids = obs[row_id, :][obs[row_id, :] != -1]
            MM_grid = np.ix_(miss_ids, miss_ids)
            MO_grid = np.ix_(miss_ids, obs_ids)
            OO_grid = np.ix_(obs_ids, obs_ids)

            S_MM, S_MO, S_OM, S_OO = sigma[MM_grid], sigma[MO_grid], sigma[MO_grid].T, sigma[OO_grid]
            # update mu params for calculate X
            M_grid = np.ix_(miss_ids)
            O_grid = np.ix_(obs_ids)
            mu_tilde[row_id] = mu[M_grid] + S_MO @ np.linalg.pinv(S_OO) @ (X[row_id, O_grid] - mu[O_grid]).flatten()

            # update X
            X[row_id, M_grid] = mu_tilde[row_id]

            # update sigma
            sigma_tilde[row_id][MM_grid] = S_MM - S_MO @ np.linalg.pinv(S_OO) @ S_OM

        # update Mu and Sigma
        mu_new = np.mean(X, axis=0)
        sigma_new = (np.cov(X.T, bias=True) + reduce(np.add, sigma_tilde.values()) / nrows)

        return mu_new, sigma_new, X

    @staticmethod
    def _converged(
            Mu: np.ndarray,
            Sigma: np.ndarray,
            Mu_new: np.ndarray,
            Sigma_new: np.ndarray,
            convergence_threshold: float
    ) -> bool:
        """Checks if the EM loop has converged.

        Args:
            Mu: np.ndarray
                The previous value of the mean.
            Sigma: np.ndarray
                The previous value of the variance.
            Mu_new: np.ndarray
                The new value of the mean.
            Sigma_new: np.ndarray
                The new value of the variance.
            convergence_threshold:
                the threshold for convergence

        Returns:
            bool: True/False if the algorithm has converged.
        """

        return (
                np.linalg.norm(Mu - Mu_new) < convergence_threshold
                and np.linalg.norm(Sigma - Sigma_new, ord=2) < convergence_threshold
        )
