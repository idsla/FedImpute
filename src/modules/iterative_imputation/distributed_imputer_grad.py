import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from src.fed_imp.sub_modules.model.logistic import LogisticRegression
from src.fed_imp.sub_modules.model.ridge import RidgeRegression
from .utils import initial_imputation, get_clip_thresholds


########################################################################################################################
# Distributed Imputer Gradient
########################################################################################################################
class DistributedFeatureImputerGrad:

	def __init__(
			self,
			missing_mask, num_cols=None, initial_strategy_cat='most_frequent', initial_strategy_num='mean',
			clip=False, batch_size=32, alpha=0.1
	):
		# missing mask and missing info
		self.missing_mask = missing_mask
		self.missing_info = {}
		for col_idx in range(self.missing_mask.shape[1]):
			self.missing_info[col_idx] = self.get_missing_info(col_idx)

		# other parameters
		self.num_cols = num_cols
		self.initial_strategy_cat = initial_strategy_cat
		self.initial_strategy_num = initial_strategy_num
		self.clip = clip
		self.estimator_placeholder = None

		# data loader and model
		self.data_loader = {}
		self.model = {}
		self.batch_size = batch_size
		self.alpha = alpha

	def initial_impute(self, X):
		# initialize
		num_cols = self.get_num_cols(X, self.num_cols)
		# imputation
		Xt = initial_imputation(X, self.initial_strategy_num, self.initial_strategy_cat, num_cols)
		return Xt

	def make_data_loader_and_model(self, X_filled, feature_indices):

		# clear data loader and model
		self.data_loader, self.model = {}, {}

		X_dims = []
		for feature_idx in feature_indices:

			# decide if col_idx is numerical column or categorical column
			num_cols = self.get_num_cols(X_filled, self.num_cols)
			if feature_idx < self.num_cols:
				estimate_type = 'numerical'
			else:
				estimate_type = 'categorical'

			# calculate correct num_cols (col_idx is numerical column then after remove it, num_cols will decrease by 1)
			if feature_idx < self.num_cols:
				num_cols = self.num_cols - 1

			# give observed part of col_idx to estimator
			row_mask = self.missing_mask[:, feature_idx]
			X_train = X_filled[~row_mask][:, np.arange(X_filled.shape[1]) != feature_idx]
			y_train = X_filled[~row_mask][:, feature_idx]

			# one hot encoding for categorical columns
			X_train_cat = X_train[:, num_cols:]
			if X_train_cat.shape[1] > 0:
				onehot_encoder = OneHotEncoder(sparse=False, max_categories=10, drop="if_binary")
				X_train_cat = onehot_encoder.fit_transform(X_train_cat)
				X_train = np.concatenate((X_train[:, :num_cols], X_train_cat), axis=1)
			else:
				X_train = X_train[:, :num_cols]

			# Convert data to PyTorch tensors
			X_train, y_train = torch.Tensor(X_train), torch.Tensor(y_train).unsqueeze(1)

			# Create PyTorch datasets and dataloaders
			train_dataset = TensorDataset(X_train, y_train)
			train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

			# Create model
			if estimate_type == 'numerical':
				model = RidgeRegression(input_dim=X_train.shape[1], alpha=self.alpha)
			else:
				model = LogisticRegression(input_dim=X_train.shape[1], alpha=self.alpha)

			# append data loader and model to dictionary
			self.data_loader[feature_idx] = train_loader
			self.model[feature_idx] = model

			X_dims.append(X_train.shape[1])

		return X_dims

	def fit_feature(self, feature_idx, global_weights, optimizer, learning_rate=0.01, local_epochs=10):

		# get model and dataloader
		dataloader = self.data_loader[feature_idx]
		model = self.model[feature_idx]

		# train model
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model.to(device)
		if global_weights is not None:
			model.output_layer.weight = torch.nn.Parameter(torch.Tensor(global_weights[:-1]).unsqueeze(0).to(device))
			model.output_layer.bias = torch.nn.Parameter(torch.Tensor([global_weights[-1]]).to(device))
		train_losses = []
		if optimizer == 'sgd':
			optimizer = optim.SGD(self.model[feature_idx].parameters(), lr=learning_rate)
		elif optimizer == 'adam':
			optimizer = optim.Adam(self.model[feature_idx].parameters(), lr=learning_rate)
		else:
			raise ValueError('Optimizer not supported')
		for epoch in range(local_epochs):
			model.train()
			total_loss, total = 0, 0
			for inputs, labels in dataloader:
				inputs, labels = inputs.to(device), labels.to(device)
				optimizer.zero_grad()
				loss = model.loss(inputs, labels)
				loss.backward()
				optimizer.step()
				total_loss += loss.item() * inputs.size(0)
				total += labels.size(0)
			train_loss = total_loss / total
			train_losses.append(train_loss)

		# get weights of model
		weights = np.append(
			model.output_layer.weight.detach().cpu().numpy(),
			model.output_layer.bias.detach().cpu().numpy()[0]
		)

		# get missing info and losses
		missing_info = self.missing_info[feature_idx]

		return weights, train_losses, missing_info

	def transform_feature(self, X, feature_idx, global_weights):

		# initialize
		num_cols = self.get_num_cols(X, self.num_cols)
		min_values, max_values = get_clip_thresholds(X, self.clip)

		# test data
		# if no missing data return X
		row_mask = self.missing_mask[:, feature_idx]
		if np.sum(row_mask) == 0:
			return X

		X_test = X[row_mask][:, np.arange(X.shape[1]) != feature_idx]

		# one hot encoding for categorical columns
		X_test_cat = X_test[:, num_cols:]
		if X_test_cat.shape[1] > 0:
			onehot_encoder = OneHotEncoder(sparse=False, max_categories=10, drop="if_binary")
			X_test_cat = onehot_encoder.fit_transform(X_test_cat)
			X_test = np.concatenate((X_test[:, :num_cols], X_test_cat), axis=1)
		else:
			X_test = X_test[:, :num_cols]

		# Convert data to PyTorch tensors
		X_test = torch.Tensor(X_test)

		# model
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = self.model[feature_idx]
		model.to(device)
		model.output_layer.weight = torch.nn.Parameter(torch.Tensor(global_weights[:-1]).unsqueeze(0).to(device))
		model.output_layer.bias = torch.nn.Parameter(torch.Tensor([global_weights[-1]]).to(device))

		model.eval()
		model.to('cpu')
		Xt = X
		with torch.no_grad():
			imputed_values = model(X_test).detach().numpy()
			imputed_values = np.clip(imputed_values, min_values[feature_idx], max_values[feature_idx])
			Xt[row_mask, feature_idx] = np.squeeze(imputed_values)

		return Xt

	@staticmethod
	def get_num_cols(X, num_cols):
		if num_cols is None:
			ret = X.shape[1]
		else:
			ret = num_cols
		return ret

	def get_missing_info(self, col_idx):

		# X train missing mask
		row_mask = self.missing_mask[:, col_idx]  # row mask
		X_train_mask = self.missing_mask[~row_mask][:, np.arange(self.missing_mask.shape[1]) != col_idx]

		sample_size = X_train_mask.shape[0]
		missing_row_pct = X_train_mask.any(axis=1).sum() / X_train_mask.shape[0]
		missing_cell_pct = X_train_mask.sum().sum() / (X_train_mask.shape[0] * X_train_mask.shape[1])
		sample_row_pct = sample_size / self.missing_mask.shape[0]

		# total pct of missing
		total_missing_cell_pct = self.missing_mask.sum().sum() / (
					self.missing_mask.shape[0] * self.missing_mask.shape[1])
		total_missing_row_pct = self.missing_mask.any(axis=1).sum() / self.missing_mask.shape[0]

		# # among all rows for training imp model, the proportion of missing part of each features
		# imp_missing_bias_weights = 1 - missing_mask_rest_features.sum(axis=0) / (~missing_row_mask).sum()
		# imp_model_feature_missing_pct.append(imp_missing_bias_weights)
		#
		# # proportion of rows used for training imp model
		# imp_model_row_pct.append((~missing_row_mask).sum() / len(missing_row_mask))

		return {
			'sample_size': sample_size,
			'sample_row_pct': sample_row_pct,
			'missing_cell_pct': missing_cell_pct,
			'missing_row_pct': missing_row_pct,
			'total_missing_cell_pct': total_missing_cell_pct,
			'total_missing_row_pct': total_missing_row_pct
		}
