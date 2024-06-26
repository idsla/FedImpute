# stdlib
from typing import Any, List, Dict, Tuple

# third party
import numpy as np
import torch
from torch import nn, optim
import torch.distributions as td
import random
import os

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


class MIWAE(nn.Module):
    """MIWAE imputation plugin

    Args:
        n_epochs: int
            Number of training iterations
        batch_size: int
            Batch size
        latent_size: int
            dimension of the latent space
        n_hidden: int
            number of hidden units
        K: int
            number of IS during training
        random_state: int
            random seed

    Reference: "MIWAE: Deep Generative Modelling and Imputation of Incomplete Data", Pierre-Alexandre Mattei,
    Jes Frellsen
    Original code: https://github.com/pamattei/miwae
    """

    def __init__(
            self,
            num_features: int,
            latent_size: int = 1,
            n_hidden: int = 1,
            seed: int = 0,
            K: int = 20,
            L: int = 1000,
    ) -> None:
        super().__init__()
        set_seed(seed)

        self.num_features = num_features
        self.n_hidden = num_features  # number of hidden units in (same for all MLPs)
        self.latent_size = num_features//2  # dimension of the latent space
        self.K = K  # number of IS during training
        self.L = L  # number of samples for imputation

        # encoder
        self.encoder = nn.Sequential(
            torch.nn.Linear(num_features, self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_hidden, self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(
                self.n_hidden, 2 * self.latent_size
            ),  # the encoder will output both the mean and the diagonal covariance
        ).to(DEVICE)

        # decoder
        self.decoder = nn.Sequential(
            torch.nn.Linear(self.latent_size, self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_hidden, self.n_hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(
                self.n_hidden, 3 * num_features
            ),
            # the decoder will output both the mean, the scale, and the number of degrees of freedoms (hence the 3*p)
        ).to(DEVICE)

        self.p_z = td.Independent(
            td.Normal(loc=torch.zeros(self.latent_size).to(DEVICE), scale=torch.ones(self.latent_size).to(DEVICE)), 1
        )

    @staticmethod
    def name() -> str:
        return "miwae"

    def init(self, seed):
        set_seed(seed)
        self.encoder.apply(weights_init)
        self.decoder.apply(weights_init)

    def compute_loss(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        x, mask = inputs  # x - data, mask - missing mask
        batch_size = x.shape[0]

        # encoder
        out_encoder = self.encoder(x)
        mu, logvar = out_encoder[..., :self.latent_size], out_encoder[..., self.latent_size:(2 * self.latent_size)]

        q_zgivenxobs = td.Independent(td.Normal(loc=mu, scale=torch.nn.Softplus()(logvar)), 1)
        zgivenx = q_zgivenxobs.rsample([self.K])  # shape (K, batch_size, latent_size)
        zgivenx_flat = zgivenx.reshape([self.K * batch_size, self.latent_size])

        # decoder
        out_decoder = self.decoder(zgivenx_flat)
        recon_x_means = out_decoder[..., :self.num_features]
        recon_x_scale = torch.nn.Softplus()(out_decoder[..., self.num_features:(2 * self.num_features)]) + 0.001
        recon_x_degree_freedom = torch.nn.Softplus()(out_decoder[..., (2 * self.num_features):]) + 3

        # compute loss
        data_flat = torch.Tensor.repeat(x, [self.K, 1]).reshape([-1, 1]).to(DEVICE)
        tiled_mask = torch.Tensor.repeat(mask, [self.K, 1]).to(DEVICE)

        all_log_pxgivenz_flat = torch.distributions.StudentT(
            loc=recon_x_means.reshape([-1, 1]),
            scale=recon_x_scale.reshape([-1, 1]),
            df=recon_x_degree_freedom.reshape([-1, 1]),
        ).log_prob(data_flat)

        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([self.K * batch_size, self.num_features])

        logpxobsgivenz = torch.sum(all_log_pxgivenz * tiled_mask, 1).reshape([self.K, batch_size])
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq, 0))

        return neg_bound, {}

    def impute(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        L = self.L
        batch_size = x.shape[0]
        p = x.shape[1]

        # encoder
        self.encoder.to(DEVICE)
        self.decoder.to(DEVICE)
        out_encoder = self.encoder(x)
        mu = out_encoder[..., : self.latent_size]
        logvar = torch.nn.Softplus()(out_encoder[..., self.latent_size: (2 * self.latent_size)])
        q_zgivenxobs = td.Independent(td.Normal(loc=mu, scale=logvar), 1)

        zgivenx = q_zgivenxobs.rsample([L])
        zgivenx_flat = zgivenx.reshape([L * batch_size, self.latent_size])

        # decoder
        out_decoder = self.decoder(zgivenx_flat)
        recon_x_means = out_decoder[..., :p]
        recon_x_scale = torch.nn.Softplus()(out_decoder[..., p: (2 * p)]) + 0.001
        recon_x_df = torch.nn.Softplus()(out_decoder[..., (2 * p): (3 * p)]) + 3

        # loss
        data_flat = torch.Tensor.repeat(x, [L, 1]).reshape([-1, 1]).to(DEVICE)
        tiledmask = torch.Tensor.repeat(mask, [L, 1]).to(DEVICE)

        all_log_pxgivenz_flat = torch.distributions.StudentT(
            loc=recon_x_means.reshape([-1, 1]),
            scale=recon_x_scale.reshape([-1, 1]),
            df=recon_x_df.reshape([-1, 1]),
        ).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L * batch_size, p])

        logpxobsgivenz = torch.sum(all_log_pxgivenz * tiledmask, 1).reshape([L, batch_size])
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        # imputation weighted samples
        xgivenz = td.Independent(
            td.StudentT(
                loc=recon_x_means,
                scale=recon_x_scale,
                df=recon_x_df,
            ),
            1
        )

        imp_weights = torch.nn.functional.softmax(
            logpxobsgivenz + logpz - logq, 0
        )  # these are w_1,....,w_L for all observations in the batch
        xms = xgivenz.sample().reshape([L, batch_size, p])
        xm = torch.einsum("ki,kij->ij", imp_weights, xms)

        # merge imputed values with observed values
        xhat = torch.clone(x)
        xhat[~mask.bool()] = xm[~mask.bool()]

        return xhat