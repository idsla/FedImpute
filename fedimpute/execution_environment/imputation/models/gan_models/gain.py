from typing import Union, Tuple, Any, List
from .....utils.reproduce_utils import set_seed
import torch
import numpy as np
import torch.nn as nn
from ..common_blocks import LinearLayers
from .....utils.nn_utils import weights_init

EPS = 1e-8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class GainModel(nn.Module):
    """The core model for GAIN Imputation.

    Args:
        dim: float
            Number of features.
        h_dim: float
            Size of the hidden layer.
        loss_alpha: int
            Hyperparameter for the generator loss.
    """

    def __init__(
            self,
            dim: int,
            h_dim: int,
            n_layers: int = 2,
            activation: str = 'relu',
            initializer: str = 'kaiming',
            loss_alpha: float = 10,
            hint_rate: float = 0.9,
            *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        # data + mask -> hidden -> hidden -> data
        self.initializer = initializer
        h_dim = h_dim
        self.generator_layer = LinearLayers(
            input_dim=dim * 2,
            output_dim=dim,
            hidden_dims=[h_dim] * n_layers,
            activation=activation,
            final_activation='sigmoid'
        ).to(DEVICE)

        # data + hints -> hidden -> hidden -> binary
        self.discriminator_layer = LinearLayers(
            input_dim=dim * 2,
            output_dim=dim,
            hidden_dims=[h_dim] * n_layers,
            activation=activation,
            final_activation='sigmoid'
        ).to(DEVICE)

        self.hint_rate = hint_rate
        self.loss_alpha = loss_alpha

    def init(self, seed):
        set_seed(seed)
        self.generator_layer.apply(lambda x: weights_init(x, self.initializer))
        self.discriminator_layer.apply(lambda x: weights_init(x, self.initializer))

    def discriminator(self, X: torch.Tensor, hints: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([X, hints], dim=1).float()
        return self.discriminator_layer(inputs)

    def generator(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([X, mask], dim=1).float()
        return self.generator_layer(inputs)

    def discr_loss(
            self, X: torch.Tensor, M: torch.Tensor, H: torch.Tensor
    ) -> torch.Tensor:
        G_sample = self.generator(X, M)
        X_hat = X * M + G_sample * (1 - M)
        D_prob = self.discriminator(X_hat, H)
        return -torch.mean(
            M * torch.log(D_prob + EPS) + (1 - M) * torch.log(1.0 - D_prob + EPS)
        )

    def gen_loss(
            self, X: torch.Tensor, M: torch.Tensor, H: torch.Tensor
    ) -> torch.Tensor:
        G_sample = self.generator(X, M)
        X_hat = X * M + G_sample * (1 - M)
        D_prob = self.discriminator(X_hat, H)

        G_loss1 = -torch.mean((1 - M) * torch.log(D_prob + EPS))
        MSE_train_loss = torch.mean((M * X - M * G_sample) ** 2) / torch.mean(M)

        return G_loss1 + self.loss_alpha * MSE_train_loss

    def impute(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        X = X.to(DEVICE)
        mask = mask.to(DEVICE)
        no, dim = X.shape
        x = X.clone()

        # Imputed data
        z = sample_Z(no, dim)
        x = mask * x + (1 - mask) * z
        imputed_data = self.generator(x, mask)

        if np.any(np.isnan(imputed_data.detach().cpu().numpy())):
            err = "The imputed result contains nan. This is a bug. Please report it on the issue tracker."
            raise RuntimeError(err)

        mask = mask.cpu().numpy()
        x = x.cpu().numpy()
        imputed_data = imputed_data.detach().cpu().numpy()
        x_imp = mask * np.nan_to_num(x) + (1 - mask) * imputed_data

        return x_imp

    def train_step(
            self, batch: Tuple[torch.Tensor, ...], batch_idx: int,
            optimizers: List[torch.optim.Optimizer], optimizer_idx: int
    ) -> tuple[float, dict]:
        x_mb, m_mb = batch

        x_mb = x_mb.to(DEVICE)
        m_mb = m_mb.to(DEVICE)
        mb_size = x_mb.size(0)
        dim = x_mb.size(1)

        # x_mb, h_mb, m_mb = sample(256, x_mb.size(0), x_mb, m_mb, x_mb.size(1), self.hint_rate)

        # if optimizer_idx == 0:
        z_mb = sample_Z(mb_size, dim)
        h_mb = sample_M(mb_size, dim, 1 - self.hint_rate)
        h_mb = m_mb * h_mb
        x_mb = m_mb * x_mb + (1 - m_mb) * z_mb

        if optimizer_idx == 0:  # discriminator
            # G_solver, D_solver = optimizers
            # D_solver.zero_grad()
            D_loss = self.discr_loss(x_mb, m_mb, h_mb)
            D_loss.backward()
            return D_loss.item(), {}
        # D_solver.step()
        else:  # generator
            # G_solver.zero_grad()
            G_loss = self.gen_loss(x_mb, m_mb, h_mb)
            G_loss.backward()
            # G_solver.step()
            return G_loss.item(), {}
        # return (D_loss.item() + G_loss.item()) / 2, {}

        #     return D_loss.item(), {}
        # else:
        #     z_mb = sample_Z(mb_size, dim)
        #     h_mb = sample_M(mb_size, dim, 1 - self.hint_rate)
        #     h_mb = m_mb * h_mb
        #     x_mb = m_mb * x_mb + (1 - m_mb) * z_mb
        #
        #     G_solver = optimizers[1]
        #     G_solver.zero_grad()
        #     G_loss = self.gen_loss(x_mb, m_mb, h_mb)
        #     G_loss.backward()
        #     G_solver.step()
        #
        #     return G_loss.item(), {}


def sample_Z(m: int, n: int) -> torch.Tensor:
    """Random sample generator for Z.

    Args:
        m: number of rows
        n: number of columns

    Returns:
        np.ndarray: generated random values
    """
    res = np.random.uniform(0.0, 0.01, size=[m, n])
    #res = np.random.normal(0.0, 0.001, size=[m, n])
    return torch.from_numpy(res).to(DEVICE)


def sample_M(m: int, n: int, p: float) -> torch.Tensor:
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


def sample_idx(m: int, n: int) -> torch.Tensor:
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


def sample(batch_size, no, X, mask, dim, hint_rate) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mb_size = min(batch_size, no)

    mb_idx = sample_idx(no, mb_size)
    x_mb = X[mb_idx, :].clone()
    m_mb = mask[mb_idx, :].clone()

    z_mb = sample_Z(mb_size, dim)
    h_mb = sample_M(mb_size, dim, 1 - hint_rate)
    h_mb = m_mb * h_mb

    x_mb = m_mb * x_mb + (1 - m_mb) * z_mb

    return x_mb, h_mb, m_mb
