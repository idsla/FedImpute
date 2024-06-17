# stdlib
from typing import Any, List, Dict, Tuple
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


# TODO: move this MM model to a separate file
class MaskNet(nn.Module):
    """
    This implements the Mask net used in notmiwae's implementation for self-masking mechanism
    """

    def __init__(self, input_dim: int):
        """
        Args:
            input_dim: Dimension of observed features.
        """
        super().__init__()
        self._device = DEVICE
        self.__input_dim = input_dim

        self.W = torch.nn.Parameter(torch.zeros([1, input_dim], device=DEVICE), requires_grad=True)
        self.b = torch.nn.Parameter(torch.zeros([1, input_dim], device=DEVICE), requires_grad=True)
        self._device = DEVICE

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (batch_size, input_dim).

        Returns:
            encoded: Encoded output tensor with shape (batch_size, input_dim)
        """
        # Run masked values through a model.
        output = -self.W * (x - self.b)
        return output


# TODO: move this MM model to a separate file
class MaskNet2(nn.Module):
    """
    This implements the Mask net that is used in notmiwae's implementation for self-masking mechanism
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        """
        Args:
            input_dim: Dimension of observed features.
        """
        super().__init__()
        self._device = DEVICE
        self.__input_dim = input_dim

        self.decoder = nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim, input_dim),
        ).to(DEVICE)
        self._device = DEVICE

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (batch_size, input_dim).

        Returns:
            encoded: Encoded output tensor with shape (batch_size, input_dim)
        """  # Run masked values through a model.
        output = self.decoder(x)
        return output


class NOTMIWAE(nn.Module):
    """
    Not-MIWAE imputation plugin
    Reference: "MIWAE: Deep Generative Modelling and Imputation of Incomplete Data", Pierre-Alexandre Mattei,
    Jes Frellsen
    Original code: https://github.com/pamattei/miwae
    """

    def __init__(
            self,
            num_features: int,
            latent_size: int = 1,
            n_hidden: int = 1,
            n_hidden_layers: int = 2,
            out_dist='studentt',
            seed: int = 0,
            K: int = 20,
            L: int = 100,
            mask_net_type: str = 'linear',
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
        self.initializer = initializer
        self.K = K  # number of IS during training
        self.L = L  # number of samples for imputation

        # mask encoder
        # self.mask_enc_dim = 0
        # self.mask_enc = nn.Sequential(
        #     torch.nn.Linear(num_features, self.n_hidden),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.n_hidden, self.mask_enc_dim)
        # )

        # encoder
        self.encoder = BaseEncoder(
            self.num_features, self.latent_size, [self.n_hidden for _ in range(self.n_hidden_layers)],
            activation=activation
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

        # mask net
        if mask_net_type == 'linear':
            self.mask_net = MaskNet(num_features).to(DEVICE)
        else:
            self.mask_net = MaskNet2(num_features).to(DEVICE)

        # prior for z
        self.p_z = td.Independent(
            td.Normal(loc=torch.zeros(self.latent_size).to(DEVICE), scale=torch.ones(self.latent_size).to(DEVICE)), 1
        )

    @staticmethod
    def name() -> str:
        return "notmiwae"

    def init(self, seed):
        set_seed(seed)
        self.encoder.apply(lambda x: weights_init(x, self.initializer))
        self.decoder.apply(lambda x: weights_init(x, self.initializer))

    def compute_loss(self, inputs: tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, Dict]:

        x, mask = inputs  # x - data, mask - missing mask
        batch_size = x.shape[0]

        # encoder and z distribution
        # mask_ = self.mask_enc(mask)
        mu, logvar = self.encoder(x)

        # q(z|x)
        q_zgivenxobs = td.Independent(td.Normal(loc=mu, scale=logvar), 1)
        zgivenx = q_zgivenxobs.rsample([self.K])
        # print(zgivenx.shape)
        zgivenx_flat = zgivenx.reshape([self.K * batch_size, self.latent_size])

        # decoder and x distribution
        out_decoder = self.decoder(zgivenx_flat)

        # data
        recon_x_means = self.decoder.l_out_mu(out_decoder)
        data = torch.Tensor.repeat(x, [self.K, 1]).to(DEVICE)
        data_flat = torch.Tensor.repeat(x, [self.K, 1]).reshape([-1, 1]).to(DEVICE)
        tiled_mask = torch.Tensor.repeat(mask, [self.K, 1]).to(DEVICE)

        # mask
        recon_x_tiled = recon_x_means * (1 - tiled_mask) + data * tiled_mask  # (K * batch_size, p)
        mask_recon = self.mask_net(recon_x_tiled).to(DEVICE).reshape([-1, 1])  # (K * batch_size*p, 1)

        # compute loss
        # p(x|z)
        all_log_pxgivenz_flat = self.decoder.dist_xgivenz(out_decoder, flat=True).log_prob(data_flat)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([self.K * batch_size, self.num_features])

        # p(m|x)
        p_mgivenx = td.Bernoulli(logits=mask_recon)  # (K * batch_size*p, 1)
        all_logp_mgivenx = p_mgivenx.log_prob(tiled_mask.reshape([-1, 1])).reshape(
            [self.K * batch_size, self.num_features])

        # final loss
        logpxobsgivenz = torch.sum(all_log_pxgivenz * tiled_mask, 1).reshape([self.K, batch_size])
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)
        logpmgivenx = torch.sum(all_logp_mgivenx, 1).reshape([self.K, batch_size])

        neg_bound = -torch.mean(torch.logsumexp(logpxobsgivenz + logpmgivenx + logpz - logq, 0))

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
        batch_size = x.shape[0]
        p = x.shape[1]

        # encoder
        self.encoder.to(DEVICE)
        self.decoder.to(DEVICE)
        self.mask_net.to(DEVICE)
        # self.mask_enc.to(DEVICE)

        mu, logvar = self.encoder(x)
        q_zgivenxobs = td.Independent(td.Normal(loc=mu, scale=logvar), 1)

        zgivenx = q_zgivenxobs.rsample([L])  # (L, batch_size, latent_size)
        zgivenx_flat = zgivenx.reshape([L * batch_size, self.latent_size])  # (L * batch_size, latent_size)

        # decoder
        out_decoder = self.decoder(zgivenx_flat)
        recon_x_means = self.decoder.l_out_mu(out_decoder)

        # ground truth data
        data = torch.Tensor.repeat(x, [L, 1]).to(DEVICE)  # (L * batch_size, p)
        data_flat = torch.Tensor.repeat(x, [L, 1]).reshape([-1, 1]).to(DEVICE)  # (L * batch_size*p, 1)
        tiled_mask = torch.Tensor.repeat(mask, [L, 1]).to(DEVICE)  # (L * batch_size, p)

        # mask net
        recon_x_tiled = recon_x_means * (1 - tiled_mask) + data * (tiled_mask)  # (L * batch_size, p)
        mask_recon = self.mask_net(recon_x_tiled).reshape([-1, 1])  # (L * batch_size*p, 1)
        p_mgivenx = td.Bernoulli(logits=mask_recon)  # (L * batch_size*p, 1)

        # loss
        all_log_pxgivenz_flat = self.decoder.dist_xgivenz(out_decoder, flat=True).log_prob(
            data_flat)  # (L * batch_size, 1)
        all_log_pxgivenz = all_log_pxgivenz_flat.reshape([L * batch_size, p])  # (L * batch_size, p)
        all_log_p_mgivenx = p_mgivenx.log_prob(tiled_mask.reshape([-1, 1])).reshape(
            [L * batch_size, p])  # (L * batch_size, p)

        logpxobsgivenz = torch.sum(all_log_pxgivenz * tiled_mask, 1).reshape([L, batch_size])  # (L, batch_size)
        logpz = self.p_z.log_prob(zgivenx)
        logq = q_zgivenxobs.log_prob(zgivenx)
        logpmgivenx = torch.sum(all_log_p_mgivenx, 1).reshape([L, batch_size])

        # imputation weighted samples
        imp_weights = torch.nn.functional.softmax(
            logpxobsgivenz + logpmgivenx + logpz - logq, 0
        )  # these are w_1,....,w_L for all observations in the batch

        xgivenz = self.decoder.imp_dist_xgivenz(out_decoder)  # torch independent to specify batch dimension
        xms = xgivenz.sample().reshape([L, batch_size, p])
        xm = torch.einsum("ki,kij->ij", imp_weights, xms)

        # merge imputed values with observed values
        xhat = torch.clone(x)
        xhat[~mask.bool()] = xm[~mask.bool()]

        return xhat
