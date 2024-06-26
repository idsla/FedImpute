from sklearn.impute import SimpleImputer
from src.modules.evaluation.evaluation import Evaluator
import numpy as np
from copy import deepcopy
from typing import Dict, Union, List, Tuple
from ..model.gain import GainModel
import torch


class ClientGAIN:
    """
    Client class for federated learning
    """

    def __init__(
            self,
            client_id,
            client_data, client_data_config: dict,
            imputation_config: dict,
            debug: bool = False,
            seed: int = 21
    ):
        ################################################################################################################
        # Data
        ################################################################################################################
        self.X_train_filled = None
        self.X_train_filled_pt = None
        self.tol = 1e-3
        self.client_id = client_id

        self.X_train, self.y_train = client_data.get('train_data')[:, :-1], client_data.get('train_data')[:, -1]
        self.X_test, self.y_test = client_data.get('test_data')[:, :-1], client_data.get('test_data')[:, -1]
        self.X_train_ms = client_data.get('train_data_ms')[:, :-1]
        self.client_data_config = client_data_config
        self.seed = seed
        self.convergency = False
        self.features_min = self.X_train.min(axis=0)
        self.features_max = self.X_train.max(axis=0)

        # get original missing mask 0-1 matrix
        self.missing_mask = np.isnan(self.X_train_ms).astype(bool)
        self.num_cols = self.client_data_config.get('num_cols', self.X_train.shape[1])
        self.task_type = self.client_data_config.get('task_type')

        # debug
        self.debug = debug
        # missingno.matrix(pd.DataFrame(self.X_train_ms))
        # plt.show()
        self.top_k_idx = None
        if self.task_type == 'classification':
            self.regression = False
        else:
            self.regression = True

        ################################################################################################################
        # imputation parameters
        ################################################################################################################
        self.initial_strategy_num = imputation_config.get('initial_strategy', 'mean')
        self.imp_clip = imputation_config.get('clip', False)
        imputation_config['imp_params']['clip'] = self.imp_clip
        self.imputation_model = GainImputer(
            imp_model_params=imputation_config['imp_model_params'],
            imp_params=imputation_config['imp_params']
        )
        self.imputation_model.initialize(self.X_train_ms, seed)

        ################################################################################################################
        # evaluation of imputation
        ################################################################################################################
        if self.task_type == 'classification':
            self.imp_evaluation_model = imputation_config.get('imp_evaluation_model', 'logistic')
            self.imp_evaluation_params = imputation_config.get('imp_evaluation_params', {})
            self.evaluation_metrics = ['imp@rmse', 'imp@w2', 'imp@sliced_ws']
        else:
            self.imp_evaluation_model = imputation_config.get('imp_evaluation_model', 'ridge')
            self.evaluation_metrics = ['imp@rmse', 'imp@w2', 'imp@sliced_ws']
            self.imp_evaluation_params = imputation_config.get('imp_evaluation_params', {})

        self.evaluator = Evaluator(
            task_type=self.task_type, metrics=self.evaluation_metrics, model=self.imp_evaluation_model,
            X_train=self.X_train, y_train=self.y_train, X_test=self.X_test, y_test=self.y_test,
            mask=self.missing_mask, seed=seed, tune_params='notune'
        )

    def fit(self, fit_instruction: dict):

        local_epoches = fit_instruction['local_epoches']

        fit_res = (
            self.imputation_model.fit(
                self.X_train_filled_pt, self.missing_mask, params={
                    'local_epochs': local_epoches
                }
            ))

        model_weights = self.imputation_model.get_imp_model_params()
        losses = fit_res
        missing_info = {"sample_size": self.get_sample_size()}

        return model_weights, losses, missing_info

    def transform(self, transform_task: str, transform_instruction: dict, global_weights):
        if transform_task == 'impute_data':
            self.X_train_filled = self.imputation_model.impute(
                self.X_train_filled_pt, self.missing_mask, features_min=self.features_min, features_max=self.features_max
            )
        elif transform_task == 'update_imp_model':
            update_weights = transform_instruction['update_weights']
            if update_weights:
                self.imputation_model.update_imp_model(global_weights)
        else:
            raise NotImplementedError

    def evaluate(
            self, evaluate_task: str, evaluate_instruction: dict, global_weights=None, update_weights: bool = True
    ):
        if evaluate_task == 'evaluate_imputation_model':
            ret = self.evaluator.evaluation_imp(self.X_train_filled)
            return ret
        elif evaluate_task == 'evaluate_prediction_model':
            return {}
        return {}

    ####################################################################################################################
    # helper functions
    ####################################################################################################################
    def get_initial_values(self):
        ret_mean, ret_max, ret_min, sample_size, ms_info = [], [], [], [], []
        for idx in range(self.X_train_ms.shape[1]):
            ret_mean.append(np.nanmean(self.X_train_ms[:, idx]))
            ret_max.append(np.nanmax(self.X_train_ms[:, idx]))
            ret_min.append(np.nanmin(self.X_train_ms[:, idx]))
            ms_info.append(self.imputation_model.get_missing_info(idx, self.missing_mask)['missing_cell_pct'])
            sample_size.append(self.imputation_model.get_missing_info(idx, self.missing_mask)['sample_row_pct'])
        return np.array(ret_mean), np.array(ret_max), np.array(ret_min), np.array(sample_size), np.array(ms_info)

    def initial_impute(self, aggregated_values):
        if aggregated_values is None:
            self.X_train_filled = self.imputation_model.initial_impute(self.X_train_ms, self.initial_strategy_num)
            self.X_train_filled_pt = self.X_train_filled.copy()
        else:
            self.X_train_filled = self.X_train_ms.copy()
            for idx in range(self.X_train_ms.shape[1]):
                indices = np.where(np.isnan(self.X_train_ms[:, idx]))[0]
                self.X_train_filled[indices, idx] = aggregated_values[idx]

            self.X_train_filled_pt = self.X_train_filled.copy()

    def get_sample_size(self):
        return self.X_train_filled.shape[0]

    def update_imp_model_minmax(self):
        self.imputation_model.features_max = self.features_max
        self.imputation_model.features_min = self.features_min


########################################################################################################################
# GAIN Imputer
########################################################################################################################
EPS = 1e-8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GainImputer:
    """GAIN Imputation for static data using Generative Adversarial Nets.
    The training steps are:
     - The generato imputes the missing components conditioned on what is actually observed, and outputs a completed vector.
     - The discriminator takes a completed vector and attempts to determine which components were actually observed and which were imputed.

    Args:
        batch_size: int
            The batch size for the training steps.
        n_epochs: int
            Number of epochs for training.
        hint_rate: float
            Percentage of additional information for the discriminator.
        loss_alpha: int
            Hyperparameter for the generator loss.

    Paper: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data Imputation using Generative Adversarial Nets," ICML, 2018.
    Original code: https://github.com/jsyoon0823/GAIN
    """

    def __init__(
            self,
            imp_model_params: dict = None,
            imp_params: dict = None,
    ) -> None:
        self.batch_size = imp_params.get("batch_size", 256)
        self.hint_rate = imp_params.get("hint_rate", 0.9)
        self.loss_alpha = imp_params.get("loss_alpha", 10)
        self.clip = imp_params.get("clip", False)
        self.lr = imp_params.get("lr", 0.001)
        self.weight_decay = imp_params.get("weight_decay", 0.0001)

        self.model_params = imp_model_params
        self.norm_parameters: Union[dict, None] = None
        self.model: Union[GainModel, None] = None
        self.norm_parameters: Union[dict, None] = None

    def initialize(self, Xmiss: np.array, seed) -> np.array:
        dim = Xmiss.shape[1]
        if 'h_dim' not in self.model_params:
            self.model_params['h_dim'] = dim
        self.model = GainModel(dim, **self.model_params)
        self.model.init(seed)

        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        for i in range(dim):
            min_val[i] = np.nanmin(Xmiss[:, i])
            max_val[i] = np.nanmax(Xmiss[:, i])
        self.norm_parameters = {"min": min_val, "max": max_val}

    @staticmethod
    def initial_impute(X, initial_strategy):

        # initialize
        X = X.copy()
        # initial imputation for numerical columns
        if initial_strategy == 'mean':
            simple_imp = SimpleImputer(strategy='mean')
            X_t = simple_imp.fit_transform(X)
        elif initial_strategy == 'median':
            simple_imp = SimpleImputer(strategy='median')
            X_t = simple_imp.fit_transform(X)
        elif initial_strategy == 'zero':
            simple_imp = SimpleImputer(strategy='constant', fill_value=0)
            X_t = simple_imp.fit_transform(X)
        else:
            raise ValueError("initial_strategy_num must be one of 'mean', 'median', 'zero'")

        return X_t

    def get_imp_model_params(self) -> dict:
        """
        Return model parameters
        """
        return deepcopy(self.model.state_dict())

    def update_imp_model(self, updated_model: dict) -> None:
        params = self.model.state_dict()
        params.update(updated_model)
        self.model.load_state_dict(params)

    def fit(self, Xmiss: np.array, mask, params: dict):
        """Train the GAIN model.

        Args:
            X: incomplete dataset.

        Returns:
            self: the updated model.
        """
        X = Xmiss.copy()
        # Parameters
        no = len(X)
        dim = len(X[0, :])
        local_epochs = params.get("local_epochs", 10)
        X = torch.from_numpy(X).float().to(DEVICE)
        mask = torch.from_numpy(~mask.copy()).float().to(DEVICE)

        # MinMaxScaler normalization
        # min_val = self.norm_parameters["min"]
        # max_val = self.norm_parameters["max"]
        #
        # for i in range(dim):
        #     X[:, i] = X[:, i] - min_val[i]
        #     X[:, i] = X[:, i] / (max_val[i] + EPS)
        #
        # # Set missing
        # mask = 1 - (1 * (np.isnan(X)))
        # mask = torch.from_numpy(mask).float().to(DEVICE)
        #
        # X = torch.from_numpy(X).to(DEVICE)
        # X = torch.nan_to_num(X)

        # Train model
        D_solver = torch.optim.Adam(
            self.model.discriminator_layer.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        G_solver = torch.optim.Adam(
            self.model.generator_layer.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        def sample() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mb_size = min(self.batch_size, no)

            mb_idx = sample_idx(no, mb_size)
            x_mb = X[mb_idx, :].clone()
            m_mb = mask[mb_idx, :].clone()

            z_mb = sample_Z(mb_size, dim)
            h_mb = sample_M(mb_size, dim, 1 - self.hint_rate)
            h_mb = m_mb * h_mb

            x_mb = m_mb * x_mb + (1 - m_mb) * z_mb

            return x_mb, h_mb, m_mb

        D_losses, G_losses = [], []
        for it in range(local_epochs):
            D_solver.zero_grad()

            x_mb, h_mb, m_mb = sample()

            D_loss = self.model.discr_loss(x_mb, m_mb, h_mb)
            D_loss.backward()
            D_solver.step()

            G_solver.zero_grad()
            x_mb, h_mb, m_mb = sample()
            G_loss = self.model.gen_loss(x_mb, m_mb, h_mb)
            G_loss.backward()
            G_solver.step()

            D_losses.append(D_loss.item())
            G_losses.append(G_loss.item())

        avg_G_loss = np.mean(G_losses)
        avg_D_loss = np.mean(D_losses)

        return {
            'g_loss': avg_G_loss,
            'd_loss': avg_D_loss
        }

    def impute(self, Xmiss: np.array, mask, features_min, features_max) -> np.array:
        """Return imputed data by trained GAIN model.

        Args:
            Xmiss: the array with missing data

        Returns:
            np.array: the array without missing data

        Raises:
            RuntimeError: if the result contains np.nans.
        """
        if self.norm_parameters is None:
            raise RuntimeError("invalid norm_parameters")
        if self.model is None:
            raise RuntimeError("Fit the model first")

        X = torch.from_numpy(Xmiss.copy()).float().to(DEVICE)
        mask = torch.from_numpy(~mask.copy()).float().to(DEVICE)
        no, dim = X.shape

        # MinMaxScaler normalization
        # X = X.cpu()
        # min_val = self.norm_parameters["min"]
        # max_val = self.norm_parameters["max"]
        # for i in range(dim):
        #     X[:, i] = X[:, i] - min_val[i]
        #     X[:, i] = X[:, i] / (max_val[i] + EPS)
        #
        # mask = 1 - (1 * (np.isnan(X)))
        # mask = mask.to(DEVICE)
        #
        # # Set missing
        # x = np.nan_to_num(X)
        x = X.clone()

        # Imputed data
        z = sample_Z(no, dim)
        x = mask * x + (1 - mask) * z

        imputed_data = self.model.generator(x, mask)

        # Renormalize
        # for i in range(dim):
        #     imputed_data[:, i] = imputed_data[:, i] * (max_val[i] + EPS)
        #     imputed_data[:, i] = imputed_data[:, i] + min_val[i]

        if np.any(np.isnan(imputed_data.detach().cpu().numpy())):
            err = "The imputed result contains nan. This is a bug. Please report it on the issue tracker."
            raise RuntimeError(err)

        mask = mask.cpu().numpy()
        imputed_data = imputed_data.detach().cpu().numpy()
        x_imp = mask * np.nan_to_num(Xmiss) + (1 - mask) * imputed_data
        if self.clip:
            for i in range(x_imp.shape[1]):
                x_imp[:, i] = np.clip(x_imp[:, i], features_min[i], features_max[i])
        return x_imp

    @staticmethod
    def get_missing_info(col_idx, missing_mask):

        # X train missing mask
        row_mask = missing_mask[:, col_idx]  # row mask
        X_train_mask = missing_mask[~row_mask][:, np.arange(missing_mask.shape[1]) != col_idx]

        sample_size = X_train_mask.shape[0]
        missing_row_pct = X_train_mask.any(axis=1).sum() / X_train_mask.shape[0]
        missing_cell_pct = X_train_mask.sum().sum() / (X_train_mask.shape[0] * X_train_mask.shape[1])
        sample_row_pct = sample_size / missing_mask.shape[0]

        # total pct of missing
        total_missing_cell_pct = missing_mask.sum().sum() / (missing_mask.shape[0] * missing_mask.shape[1])
        total_missing_row_pct = missing_mask.any(axis=1).sum() / missing_mask.shape[0]

        return {
            'sample_size': sample_size,
            'sample_row_pct': sample_row_pct,
            'missing_cell_pct': missing_cell_pct,
            'missing_row_pct': missing_row_pct,
            'total_missing_cell_pct': total_missing_cell_pct,
            'total_missing_row_pct': total_missing_row_pct
        }


def sample_Z(m: int, n: int) -> np.ndarray:
    """Random sample generator for Z.

    Args:
        m: number of rows
        n: number of columns

    Returns:
        np.ndarray: generated random values
    """
    res = np.random.uniform(0.0, 0.01, size=[m, n])
    return torch.from_numpy(res).to(DEVICE)


def sample_M(m: int, n: int, p: float) -> np.ndarray:
    """Hint Vector Generation

    Args:
        m: number of rows
        n: number of columns
        p: hint rate

    Returns:
        np.ndarray: generated random values
    """
    unif_prob = np.random.uniform(0.0, 1.0, size=[m, n])
    M = unif_prob > p
    M = 1.0 * M

    return torch.from_numpy(M).to(DEVICE)


def sample_idx(m: int, n: int) -> np.ndarray:
    """Mini-batch generation

    Args:
        m: number of rows
        n: number of columns

    Returns:
        np.ndarray: generated random indices
    """
    idx = np.random.permutation(m)
    idx = idx[:n]
    return idx