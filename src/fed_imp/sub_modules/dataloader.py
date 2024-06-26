import torch
from torch.utils.data import Dataset, TensorDataset
from sklearn.preprocessing import OneHotEncoder


def construct_tensor_dataset(X, y):
	dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
	return dataset
