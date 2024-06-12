# stdlib
from typing import Any, List, Dict, Tuple

# third party
import numpy as np
import torch
from torch import nn, optim
import torch.distributions as td

# hyperimpute absolute
from .....utils.reproduce_utils import set_seed
from .decoder import GaussianDecoder, BernoulliDecoder, StudentTDecoder
from .encoder import BaseEncoder
from .....utils.nn_utils import weights_init

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


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
            n_hidden: int = 16,
            n_hidden_layers: int = 2,
            seed: int = 0,
            out_dist='studentt',
            K: int = 20,
            L: int = 1000,
            activation='tanh',
            initializer='xavier'
    ) -> None:

        super().__init__()
        set_seed(seed)

        # parameters
        self.num_features = num_features
        self.n_hidden = n_hidden  # number of hidden units in (same for all MLPs)
        self.n_hidden_layers = n_hidden_layers  # number of hidden layers in (same for all MLPs)
        self.latent_size = latent_size  # dimension of the latent space
        self.K = K  # number of IS during training
        self.L = L  # number of samples for imputation
        self.initializer = initializer

        # encoder
        self.encoder = BaseEncoder(
            self.num_features, self.latent_size, [self.n_hidden for _ in range(self.n_hidden_layers)],
            activation = activation
        ).to(DEVICE)

        # decoder
        self.out_dist = out_dist
        if out_dist == 'studentt':
            self.decoder = StudentTDecoder(
                self.latent_size, self.num_features, [self.n_hidden for _ in range(self.n_hidden_layers)],
                activation=activation
            )
        elif out_dist == 'gaussian':
            self.decoder = GaussianDecoder(
                self.latent_size, self.num_features, [self.n_hidden for _ in range(self.n_hidden_layers)],
                activation=activation
            )
        elif out_dist == 'bernoulli':
            self.decoder = BernoulliDecoder(
                self.latent_size, self.num_features, [self.n_hidden for _ in range(self.n_hidden_layers)],
                activation=activation
            )
        else:
            raise ValueError("Invalid output distribution")

        self.decoder = self.decoder.to(DEVICE)

        # prior for z
        self.p_z = td.Independent(
            td.Normal(loc=torch.zeros(self.latent_size).to(DEVICE), scale=torch.ones(self.latent_size).to(DEVICE)),
            1
        )

    @staticmethod
    def name() -> str:
        return "miwae"

    def init(self, seed):
        set_seed(seed)
        self.encoder.apply(lambda x: weights_init(x, self.initializer))
        self.decoder.apply(lambda x: weights_init(x, self.initializer))

    def compute_loss(self, inputs: tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, Dict]:
        x, mask = inputs  # x - data, mask - missing mask
        batch_size = x.shape[0]

        # encoder
        mu, logvar = self.encoder(x)

        q_zgivenxobs = td.Independent(td.Normal(loc=mu, scale=logvar), 1)
        zgivenx = q_zgivenxobs.rsample([self.K])  # shape (K, batch_size, latent_size)
        zgivenx_flat = zgivenx.reshape([self.K * batch_size, self.latent_size])

        # decoder
        out_decoder = self.decoder(zgivenx_flat)
        # recon_x_means = self.decoder.l_out_mu(out_decoder)

        # compute loss
        data_flat = torch.Tensor.repeat(x, [self.K, 1]).reshape([-1, 1]).to(DEVICE)
        tiled_mask = torch.Tensor.repeat(mask, [self.K, 1]).to(DEVICE)

        # p(x|z)
        all_log_pxgivenz_flat = self.decoder.dist_xgivenz(out_decoder, flat=True).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([self.K * batch_size, self.num_features])
        logpxobsgivenz = torch.sum(all_log_pxgivenz * tiled_mask, 1).reshape([self.K, batch_size])

        # p(z) and q(z|x)
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpz - logq, 0))

        return neg_bound, {}

    def train_step(
            self, batch: Tuple[torch.Tensor, ...], batch_idx: int,
            optimizers: List[torch.optim.Optimizer], optimizer_idx: int
    ) -> tuple[float, dict]:

        batch = tuple(item.to(DEVICE) for item in batch)
        loss, train_res_dict = self.compute_loss(batch)
        loss.backward()
        #grad_scaler.scale(loss).backward()

        return loss.item(), {}

    def impute(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        L = self.L
        batch_size = x.shape[0]
        p = x.shape[1]

        # encoder
        self.encoder.to(DEVICE)
        self.decoder.to(DEVICE)
        mu, logvar = self.encoder(x)
        q_zgivenxobs = td.Independent(td.Normal(loc=mu, scale=logvar), 1)

        zgivenx = q_zgivenxobs.rsample([L])
        zgivenx_flat = zgivenx.reshape([L * batch_size, self.latent_size])

        # decoder
        out_decoder = self.decoder(zgivenx_flat)
        # recon_x_means = self.decoder.l_out_mu(out_decoder)

        # loss
        data_flat = torch.Tensor.repeat(x, [L, 1]).reshape([-1, 1]).to(DEVICE)
        tiledmask = torch.Tensor.repeat(mask, [L, 1]).to(DEVICE)

        all_log_pxgivenz_flat = self.decoder.dist_xgivenz(out_decoder, flat=True).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L * batch_size, p])

        logpxobsgivenz = torch.sum(all_log_pxgivenz * tiledmask, 1).reshape([L, batch_size])
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)

        # imputation weighted samples
        imp_weights = torch.nn.functional.softmax(
            logpxobsgivenz + logpz - logq, 0
        )  # these are w_1,....,w_L for all observations in the batch

        xgivenz = self.decoder.imp_dist_xgivenz(out_decoder)
        xms = xgivenz.sample().reshape([L, batch_size, p])
        xm = torch.einsum("ki,kij->ij", imp_weights, xms)

        # merge imputed values with observed values
        xhat = torch.clone(x)
        xhat[~mask.bool()] = xm[~mask.bool()]

        return xhat
