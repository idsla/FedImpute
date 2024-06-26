import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoNNReg(nn.Module):
	def __init__(self, in_features, num_hiddens=128):
		super(TwoNNReg, self).__init__()
		self.activation = nn.ReLU(True)
		self.fc1 = nn.Linear(in_features=in_features, out_features=num_hiddens, bias=True)
		self.fc2 = nn.Linear(in_features=num_hiddens, out_features=num_hiddens, bias=True)
		self.fc3 = nn.Linear(in_features=num_hiddens, out_features=1, bias=True)

	def forward(self, x):
		if x.ndim == 4:
			x = x.view(x.size(0), -1)
		x = self.activation(self.fc1(x))
		x = self.activation(self.fc2(x))
		x = self.fc3(x)
		return x
