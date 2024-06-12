import torch
import torch.nn as nn
from ..common_blocks import LinearLayers


class BaseEncoder(nn.Module):
    def __init__(
            self,
            input_dim,
            latent_dim,
            hidden_dims,
            dropout_rate=0.0,
            normalization=None,
            activation='tanh'
    ):
        super(BaseEncoder, self).__init__()
        self.hidden_layers = LinearLayers(
            input_dim, hidden_dims[-1], hidden_dims[:-1], dropout_rate, normalization, activation
        )
        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.log_var_layer = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x):
        x = self.hidden_layers(x)
        mu = self.mu_layer(x)
        log_var = torch.nn.Softplus()(self.log_var_layer(x))  # TODO: Check whether we need this softplus
        return mu, log_var

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z