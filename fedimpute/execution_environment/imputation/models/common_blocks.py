from typing import Type, Any

import torch

import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(layer: Any) -> None:
    if type(layer) == nn.Linear:
        torch.nn.init.orthogonal_(layer.weight)


########################################################################################################################
# Define the alternative initialization function
def alternative_initialization(module, init_method: str) -> None:
    if isinstance(module, nn.Linear):
        if init_method == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        elif init_method == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif init_method == "uniform":
            nn.init.uniform_(module.weight)
        elif init_method == "normal":
            nn.init.normal_(module.weight)
        else:
            return
        nn.init.zeros_(module.bias)


########################################################################################################################
# Residual Block
class ResBlock(nn.Module):
    """
    Wraps an nn.Module, adding a skip connection to it.
    """

    def __init__(self, block: Type[nn.Module]):
        """
        Args:
            block: module to which skip connection will be added. The input dimension must match the output dimension.
        """
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


########################################################################################################################
# Linear Layers
class LinearLayers(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hidden_dims,
            dropout_rate=0.0,
            normalization=None,
            activation='relu',
            final_activation=None,
            init_method='default'
    ):
        super(LinearLayers, self).__init__()

        modules = []
        prev_dim = input_dim

        for dim in hidden_dims:
            # Add Linear layer without immediate initialization
            linear = nn.Linear(prev_dim, dim)
            modules.append(linear)

            # Add Normalization layer
            if normalization is not None:
                if normalization == 'batch':
                    modules.append(nn.BatchNorm1d(dim))
                elif normalization == 'layer':
                    modules.append(nn.LayerNorm(dim))
                else:
                    raise ValueError(f'Unknown normalization type: {normalization}')

            # Add Activation layer
            if activation == 'relu':
                modules.append(nn.ReLU())
            elif activation == 'tanh':
                modules.append(nn.Tanh())
            else:
                raise ValueError(f'Unknown activation type: {activation}')

            # Add Dropout layer
            if dropout_rate > 0:
                modules.append(nn.Dropout(dropout_rate))

            prev_dim = dim

        # Final Linear layer
        modules.append(nn.Linear(hidden_dims[-1], output_dim))

        # Optional: Final Activation layer
        if final_activation is not None:
            if final_activation == 'sigmoid':
                modules.append(nn.Sigmoid())
            elif final_activation == 'softmax':
                modules.append(nn.Softmax(dim=-1))

        # Combine all modules into a sequential container
        self.model = nn.Sequential(*modules)

        # Apply the alternative initialization
        if init_method != 'default':
            self.model.apply(lambda module: alternative_initialization(module, init_method))

    def forward(self, x):
        return self.model(x)
