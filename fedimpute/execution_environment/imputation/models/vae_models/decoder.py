import torch
from ..common_blocks import LinearLayers
import torch.nn as nn
import torch.distributions as td


class GaussianDecoder(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dims,
            dropout_rate=0.0,
            normalization=None,
            activation='tanh'
    ):
        super(GaussianDecoder, self).__init__()
        self.hidden_layers = LinearLayers(
            input_dim, hidden_dims[-1], hidden_dims[:-1], dropout_rate, normalization, activation
        )

        self.mu_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.std_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Softplus()
        )

    def forward(self, x):
        x = self.hidden_layers(x)
        mu = self.mu_layer(x)
        std = self.std_layer(x)
        return mu, std

    @staticmethod
    def dist_xgivenz(decoder_out, flat=True):
        mu, std = decoder_out
        if flat:
            return torch.distributions.Normal(
                loc=mu.reshape([-1, 1]),
                scale=std.reshape([-1, 1])
            )
        else:
            return torch.distributions.Normal(
                loc=mu,
                scale=std
            )

    @staticmethod
    def imp_dist_xgivenz(decoder_out):
        mu, std = decoder_out
        return td.Independent(
            td.Normal(loc=mu, scale=std),
            1
        )

    @staticmethod
    def l_out_mu(decoder_out):
        mu = decoder_out[0]
        return mu


class BernoulliDecoder(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dims,
            dropout_rate=0.0,
            normalization=None,
            activation='tanh'
    ):
        super(BernoulliDecoder, self).__init__()
        self.hidden_layers = LinearLayers(
            input_dim, hidden_dims[-1], hidden_dims[:-1], dropout_rate, normalization, activation
        )
        self.logits_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = self.hidden_layers(x)
        logits = self.logits_layer(x)
        return logits

    @staticmethod
    def dist_xgivenz(decoder_out, flat=True):
        logits = decoder_out
        if flat:
            return td.Bernoulli(logits=logits.reshape([-1, 1]))
        else:
            return td.Bernoulli(logits=logits)

    @staticmethod
    def imp_dist_xgivenz(decoder_out):
        logits = decoder_out
        return td.Independent(
            td.Bernoulli(logits=logits),
            1
        )

    @staticmethod
    def l_out_mu(decoder_out):
        logits = decoder_out
        return torch.sigmoid(logits)


class StudentTDecoder(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dims,
            dropout_rate=0.0,
            normalization=None,
            activation='tanh'
    ):
        super(StudentTDecoder, self).__init__()
        self.hidden_layers = LinearLayers(
            input_dim, hidden_dims[-1], hidden_dims[:-1], dropout_rate, normalization, activation
        )
        self.mu_layer = nn.Linear(hidden_dims[-1], output_dim)
        self.std_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Softplus()
        )
        self.df_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.Softplus(),
            # nn.Lambda(lambda x: x + 2)  # Ensure df > 2
        )

    def forward(self, x):
        x = self.hidden_layers(x)
        mu = self.mu_layer(x)
        std = self.std_layer(x) + 0.001
        df = self.df_layer(x) + 3
        return mu, std, df

    @staticmethod
    def dist_xgivenz(decoder_out, flat=True):
        mu, std, df = decoder_out
        if flat:
            return td.StudentT(
                loc=mu.reshape([-1, 1]),
                scale=std.reshape([-1, 1]),
                df=df.reshape([-1, 1])
            )
        else:
            return td.StudentT(
                loc=mu,
                scale=std,
                df=df
            )

    @staticmethod
    def imp_dist_xgivenz(decoder_out):
        mu, std, df = decoder_out
        return td.Independent(
            td.StudentT(
                loc=mu,
                scale=std,
                df=df
            ),
            1
        )

    @staticmethod
    def l_out_mu(decoder_out):
        mu = decoder_out[0]
        return mu
