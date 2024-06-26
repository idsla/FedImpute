import os
from copy import deepcopy
import random
from typing import Union, Tuple, Any

import torch
import numpy as np
import torch.nn as nn

EPS = 1e-8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.use_deterministic_algorithms(True)


def weights_init(layer: Any) -> None:
    if type(layer) == nn.Linear:
        torch.nn.init.orthogonal_(layer.weight)


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

    def __init__(self, dim: int, h_dim: int, loss_alpha: float = 10, *args, **kwargs) -> None:

        super().__init__(*args, **kwargs)
        self.generator_layer = nn.Sequential(           # data + mask -> hidden -> hidden -> data
            nn.Linear(dim * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, dim),
            nn.Sigmoid(),
        ).to(DEVICE)

        self.discriminator_layer = nn.Sequential(      # data + hints -> hidden -> hidden -> binary
            nn.Linear(dim * 2, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, dim),
            nn.Sigmoid(),
        ).to(DEVICE)

        self.loss_alpha = loss_alpha

    def init(self, seed):
        set_seed(seed)
        self.generator_layer.apply(weights_init)
        self.discriminator_layer.apply(weights_init)

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