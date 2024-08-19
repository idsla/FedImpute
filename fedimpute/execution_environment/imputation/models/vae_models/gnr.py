from typing import Any, List, Dict, Tuple
import numpy as np
import torch
from torch import nn, optim
import torch.distributions as td

# hyperimpute absolute
from .....utils.reproduce_utils import set_seed
from .encoder import BaseEncoder
from ..common_blocks import LinearLayers
from .....utils.nn_utils import weights_init

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FusionDecoder(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dims,
            dropout_rate=0.0,
            normalization=None,
            activation='tanh'
    ):
        super(FusionDecoder, self).__init__()

        # Check if there are hidden layers to add
        self.hidden_layers = LinearLayers(
            input_dim, hidden_dims[-1], hidden_dims[:-1], dropout_rate, normalization, activation
        )

        # reconstruction x
        self.mu_layer = nn.Linear(
            hidden_dims[-1], output_dim
        )
        self.std_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim), nn.Softplus()
        )

        # reconstruction mask
        self.mask_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim), nn.Softplus(),
            # nn.Lambda(lambda x: x + 2)  # Ensure df > 2
        )

    def forward(self, x):
        x = self.hidden_layers(x)
        mu = self.mu_layer(x)
        std = self.std_layer(x) + 0.001
        mask = self.mask_layer(x)
        return mu, std, mask


class GNR(nn.Module):

    def __init__(
            self,
            num_features: int,
            latent_size: int = 1,
            n_hidden: int = 1,
            n_hidden_layers: int = 2,
            seed: int = 0,
            K: int = 20,
            loss_ceof=10,
            mr_loss_coef: bool = True,
            L: int = 1000,
            activation='tanh',
            initializer='xavier',
    ) -> None:
        super().__init__()
        set_seed(seed)

        self.num_features = num_features
        self.n_hidden = n_hidden  # number of hidden units in (same for all MLPs)
        self.latent_size = latent_size  # dimension of the latent space
        self.K = K  # number of IS during training
        self.L = L  # number of samples for imputation
        self.n_hidden_layers = n_hidden_layers  # number of hidden layers in (same for all MLPs)
        self.loss_coef = loss_ceof
        self.mr_loss_coef = mr_loss_coef
        self.initializer = initializer

        # encoder
        self.encoder = BaseEncoder(
            self.num_features, self.latent_size, [self.n_hidden for _ in range(self.n_hidden_layers)],
            activation=activation
        ).to(DEVICE)

        # fusion decoder
        self.decoder = FusionDecoder(
            self.latent_size, self.num_features, [self.n_hidden for _ in range(self.n_hidden_layers)],
            activation=activation
        ).to(DEVICE)

        # prior
        self.p_z = td.Independent(
            td.Normal(loc=torch.zeros(self.latent_size).to(DEVICE), scale=torch.ones(self.latent_size).to(DEVICE)),
            1
        )

    @staticmethod
    def name() -> str:
        return "gnr"

    def init(self, seed):
        set_seed(seed)
        self.encoder.apply(lambda x: weights_init(x, self.initializer))
        self.decoder.apply(lambda x: weights_init(x, self.initializer))

    def compute_loss(self, inputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, Dict]:
        x, mask = inputs  # x - data, mask - missing mask
        missing_ratio = (1 - mask).sum() / (mask.shape[0] * mask.shape[1]) + 1e-6

        # encoder
        q_mu, q_logvar = self.encoder(x)  # (batch_size, latent_size)

        q_std = torch.sqrt(torch.exp(q_logvar))
        q_zgivenx = td.Independent(
            td.Normal(loc=q_mu, scale=q_std), 1  # todo check consistency of std and logvar
        )
        l_z = q_zgivenx.rsample([self.K])  # (K, batch_size, latent_size)
        l_z = torch.transpose(l_z, 0, 1)  # (batch_size, K, latent_size)

        # decoder
        mu, std, mask_pred = self.decoder(l_z)  # (batch_size, K, num_features)

        ################################################################################################################
        # Loss
        # p(x|z)
        p_x_given_z = td.Normal(loc=mu, scale=std)
        log_px_given_z = torch.sum(
            torch.unsqueeze(mask, 1) * p_x_given_z.log_prob(torch.unsqueeze(x, 1)),  # (batch_size, 1, num_features)
            dim=-1
        )  # (batch_size, K)

        # p(m|x)
        p_m_given_z = td.Bernoulli(logits=mask_pred)
        log_pm_given_z = torch.sum(
            p_m_given_z.log_prob(torch.unsqueeze(mask, dim=1)),  # (batch_size, 1, num_features)
            dim=-1
        )  # (batch_size, K)

        # q(z|x)
        # q_z_given_x2 = td.Independent(
        #     td.Normal(loc=torch.unsqueeze(q_mu, dim=1), scale=torch.unsqueeze(q_std, dim=1)),
        #     1
        # )
        log_qz_given_x = q_zgivenx.log_prob(l_z)  # (batch_size, K)

        # p(z)
        log_pz = self.p_z.log_prob(l_z)  # (batch_size, K)

        # final loss
        if self.mr_loss_coef:
            l_w = log_px_given_z + log_pm_given_z * self.loss_coef / missing_ratio.item() + log_pz - log_qz_given_x
        else:
            l_w = log_px_given_z + self.loss_coef * log_pm_given_z + log_pz - log_qz_given_x

        # (batch_size, K)
        l_w = torch.logsumexp(l_w, dim=1) - np.log(float(self.K))  # (batch_size,)
        neg_bound = -torch.mean(l_w, dim=-1)  # (1,)

        return neg_bound, {}

    def train_step(
            self, batch: Tuple[torch.Tensor, ...], batch_idx: int,
            optimizers: List[torch.optim.Optimizer], optimizer_idx: int
    ) -> tuple[float, dict]:
        batch = tuple(item.to(DEVICE) for item in batch)
        loss, train_res_dict = self.compute_loss(batch)
        loss.backward()

        return loss.item(), {}

    def impute(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        L = self.L
        missing_ratio = (1 - mask).sum() / (mask.shape[0] * mask.shape[1]) + 1e-6

        # encoder
        self.encoder.to(DEVICE)
        self.decoder.to(DEVICE)
        q_mu, q_logvar = self.encoder(x)
        q_std = torch.sqrt(torch.exp(q_logvar))

        q_zgivenx = td.Independent(td.Normal(loc=q_mu, scale=q_std),1)  # todo check consistency of std and logvar
        l_z = q_zgivenx.rsample([L])  # (L, batch_size, latent_size)
        l_z = torch.transpose(l_z, 0, 1)  # (batch_size, L, latent_size)

        # decoder
        mu, std, mask_pred = self.decoder(l_z)  # (batch_size, K, num_features)

        ################################################################################################################
        # Loss
        # p(x|z)
        p_x_given_z = td.Normal(loc=mu, scale=std)
        log_px_given_z = torch.sum(
            # (batch_size, L, num_features)
            torch.unsqueeze(mask, 1) * p_x_given_z.log_prob(torch.unsqueeze(x, 1)), dim=-1
        )  # (batch_size, L)

        # p(m|x)
        p_m_given_z = td.Bernoulli(logits=mask_pred)
        log_pm_given_z = torch.sum(
            p_m_given_z.log_prob(torch.unsqueeze(mask, dim=1)), dim=-1  # (batch_size, L, num_features)
        )  # (batch_size, L)

        # q(z|x)
        # q_z_given_x2 = td.Independent(
        #     td.Normal(loc=torch.unsqueeze(q_mu, dim=1), scale=torch.unsqueeze(q_std, dim=1)), 1
        # )
        l_z_reshaped = l_z.view(-1, self.latent_size)  # (batch_size*L, latent size)
        log_qz_given_x = q_zgivenx.log_prob(l_z_reshaped) # (batch_size, L)


        # p(z)
        log_pz = self.p_z.log_prob(l_z)  # (batch_size, L)

        # final loss
        if self.mr_loss_coef:
            l_w = log_px_given_z + log_pm_given_z * self.loss_coef / missing_ratio + log_pz - log_qz_given_x
        else:
            l_w = log_px_given_z + self.loss_coef * log_pm_given_z + log_pz - log_qz_given_x

        # (batch_size, L)
        l_w = torch.nn.Softmax(dim=1)(l_w)  # (batch_size, L)

        # imputation weighted samples
        l_x_given_z = td.Normal(loc=mu, scale=std).rsample([L])  # (L, batch_size, num_features)
        l_x_given_z = torch.transpose(l_x_given_z, 0, 1)  # (batch_size, L, num_features)

        # imputation
        xm = torch.einsum("ki,kij->ij", l_w, l_x_given_z)  # (batch_size, num_features)
        xhat = torch.clone(x)
        xhat[~mask.bool()] = xm[~mask.bool()]

        return xhat
