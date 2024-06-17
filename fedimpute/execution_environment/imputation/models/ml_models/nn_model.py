from typing import List, Dict, Tuple

import torch.nn as nn
import torch

from .....utils.reproduce_utils import set_seed
from ..common_blocks import weights_init
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TwoNNRegressor(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_rate=0.5):
        super(TwoNNRegressor, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

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


class TwoNNClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_rate=0.5):
        super(TwoNNClassifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

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

