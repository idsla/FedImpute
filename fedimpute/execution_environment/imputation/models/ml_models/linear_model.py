from typing import List, Tuple, Dict

import torch.nn as nn
import torch

from .....utils.reproduce_utils import set_seed
from ..common_blocks import weights_init
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RidgeRegression(nn.Module):

    def __init__(self, input_dim):
        super(RidgeRegression, self).__init__()
        # self.w = nn.Parameter(torch.zeros(input_dim, 1))
        # self.b = nn.Parameter(torch.zeros(1))
        self.linear = nn.Linear(input_dim, 1)

    def init(self, seed):
        set_seed(seed)
        self.linear.apply(weights_init)

    def forward(self, x):
        return self.linear(x)

    def compute_loss(self, inputs: tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, Dict]:
        x, y_true = inputs
        y_pred = self.forward(x)
        return torch.nn.MSELoss()(y_pred, y_true), {}

    def train_step(
            self, batch: Tuple[torch.Tensor, ...], batch_idx: int,
            optimizers: List[torch.optim.Optimizer], optimizer_idx: int
    ) -> tuple[float, dict]:
        optimizer = optimizers[0]
        batch = tuple(item.to(DEVICE) for item in batch)
        optimizer.zero_grad()
        loss, train_res_dict = self.compute_loss(batch)
        loss.backward()
        optimizer.step()

        return loss.item(), {}


class Logistic(nn.Module):

    def __init__(self, input_dim, C=0.1):
        super(Logistic, self).__init__()
        self.C = C
        self.linear = nn.Linear(input_dim, 1)

    def init(self, seed):
        set_seed(seed)
        self.linear.apply(weights_init)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def compute_loss(self, inputs: tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, Dict]:
        x, y_true = inputs
        y_pred = self.forward(x)
        loss = nn.BCELoss()(y_pred, y_true.view_as(y_pred))  # Ensure y_true is the same shape as y_pred
        return loss, {}

    def train_step(
            self, batch: Tuple[torch.Tensor, ...], batch_idx: int,
            optimizers: List[torch.optim.Optimizer], optimizer_idx: int
    ) -> tuple[float, dict]:
        optimizer = optimizers[0]
        batch = tuple(item.to(DEVICE) for item in batch)
        optimizer.zero_grad()
        loss, train_res_dict = self.compute_loss(batch)
        loss.backward()
        optimizer.step()

        return loss.item(), {}
