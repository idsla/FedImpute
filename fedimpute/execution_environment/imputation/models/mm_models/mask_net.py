import torch
from torch import nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LinearMaskNet(nn.Module):
    """
    This implements the Mask net that is used in notmiwae's implementation for self-masking mechanism
    """

    def __init__(self, input_dim: int, latent_dim: int, latent_connection: bool = False):
        """
        Args:
            input_dim: Dimension of observed features.
            device: torch device to use.
        """
        super().__init__()
        self._device = DEVICE
        if latent_connection:
            self.__input_dim = input_dim + latent_dim
        self.__input_dim = input_dim

        self.W = torch.nn.Parameter(torch.zeros([1, self.__input_dim], device=DEVICE), requires_grad=True)
        self.b = torch.nn.Parameter(torch.zeros([1, self.__input_dim], device=DEVICE), requires_grad=True)
        self._device = DEVICE

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (batch_size, input_dim).

        Returns:
            encoded: Encoded output tensor with shape (batch_size, input_dim)
        """  # Run masked values through model.
        output = -self.W * (x - self.b)
        return output


class NNMaskNet(nn.Module):
    """
    This implements the Mask net that is used in notmiwae's implementation for self-masking mechanism
    """

    def __init__(self, input_dim: int, latent_dim: int, latent_connection: bool = False):
        """
        Args:
            input_dim: Dimension of observed features.
            device: torch device to use.
        """
        super().__init__()
        self._device = DEVICE
        if latent_connection:
            self.__input_dim = input_dim + latent_dim
        self.__input_dim = input_dim

        self.W = torch.nn.Parameter(torch.zeros([1, self.__input_dim], device=DEVICE), requires_grad=True)
        self.b = torch.nn.Parameter(torch.zeros([1, self.__input_dim], device=DEVICE), requires_grad=True)
        self._device = DEVICE

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (batch_size, input_dim).

        Returns:
            encoded: Encoded output tensor with shape (batch_size, input_dim)
        """  # Run masked values through model.
        output = -self.W * (x - self.b)
        return output
