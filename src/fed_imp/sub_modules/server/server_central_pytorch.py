import os
import random
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from src.fed_imp.sub_modules.server.base_server import ServerBase
from src.fed_imp.sub_modules.strategy.strategy_imp import StrategyImputation
from src.fed_imp.sub_modules.client.client import Client
from typing import Dict, List
from src.utils import set_seed
from src.fed_imp.sub_modules.model.TwoNN import TwoNN
from src.fed_imp.sub_modules.model.logistic import LogisticRegression
from src.fed_imp.sub_modules.model.utils import EarlyStopping
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def client_fit(args):
	client, client_id, col_idx = args
	weight, loss, missing_info, projection_matrix, ms_coef = client.fit(
		fit_task='fit_imputation_model', fit_instruction={'feature_idx': col_idx}
	)
	return client_id, weight, loss, missing_info, projection_matrix, ms_coef, client


def process_args(args):
	return client_fit(*args)


class ServerCentralPytorch(ServerBase):

	def __init__(
			self,
			clients: Dict[int, Client], strategy_imp: StrategyImputation, server_config: dict,
			pred_config: dict, test_data: np.ndarray = None, seed: int = 21, base_model='twonn',
			track=False, run_prediction=True, persist_data=False
	):

		super().__init__(
			clients, strategy_imp, server_config, pred_config, test_data, seed, track, run_prediction, persist_data
		)

		self.clients = clients
		self.strategy_imp = strategy_imp
		self.max_workers = None
		self.config = server_config
		self.seed = seed
		self.base_model = base_model

		# imputation parameters
		self.num_rounds_imp = server_config.get('imp_round', 30)
		self.imp_model_fit_mode = server_config.get('model_fit_mode', "one_shot")

		# group clients
		self.client_groups = {}

		###########################################################################
		# Prediction model
		###########################################################################
		# model
		self.pred_model_params = pred_config.get(
			'model_params', {
				"num_hiddens": 64,
				"input_feature_dim": test_data.shape[1] - 1,
				'output_classes_dim': len(np.unique(test_data[:, -1]))
			}
			)

		self.pred_training_params = pred_config.get(
			'training_params', {
				"batch_size": 128,
				"learning_rate": 0.001,
				"weight_decay": 1e-4,
				"pred_round": 300
			}
			)

		# test data
		self.test_data = test_data

	def prediction(self, seeds=None):

		###############################################################################################
		# Collecting client imputed datasets
		###############################################################################################
		# setup work in client side
		if seeds is None:
			seeds = [42, 44]

		validate_data, train_data = [], []
		for client_id, client in self.clients.items():
			client.local_dataset()
			validate_data.append(client.val_data)
			train_data.append(client.train_data)

		dataset = np.concatenate(train_data, axis=0)
		X_train, y_train = dataset[:, :-1], dataset[:, -1].astype(int)
		X_test, y_test = self.test_data[:, :-1], self.test_data[:, -1].astype(int)

		################################################################################################
		# Training prediction model
		################################################################################################
		hidden_size = self.pred_model_params['num_hiddens']
		train_epochs = self.pred_training_params['pred_round']
		batch_size = self.pred_training_params['batch_size']
		learning_rate = self.pred_training_params['learning_rate']
		weight_decay = self.pred_training_params['weight_decay']

		train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
		train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

		# centralized evaluation
		accus, f1s = [], []
		for s in seeds:
			set_seed(s)
			early_stopping = EarlyStopping()

			# Model. loss and optimizer
			if self.base_model == 'twonn':
				model = TwoNN(
					in_features=self.pred_model_params['input_feature_dim'],
					num_classes=self.pred_model_params['output_classes_dim'],
					num_hiddens=hidden_size
				)
			elif self.base_model == 'lr':
				model = LogisticRegression(
					in_features=self.pred_model_params['input_feature_dim'],
					num_classes=self.pred_model_params['output_classes_dim']
				)
			else:
				raise ValueError('base model not found')

			criterion = torch.nn.CrossEntropyLoss()
			optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

			# Train the model
			train_losses, val_losses = [], []
			for epoch in range(train_epochs):
				model.to(DEVICE)
				model.train()  # prep model for training

				train_running_loss = 0.0
				counter = 0
				for i, (X_train, y_train) in enumerate(train_dataloader):
					counter += 1
					# Move tensors to the configured device
					X_train, y_train = X_train.to(DEVICE), y_train.to(DEVICE)

					# Forward pass
					outputs = model(X_train)
					loss = criterion(outputs, y_train)

					# Backward and optimize
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					train_running_loss += loss.item()

					if DEVICE == 'cuda':
						torch.cuda.empty_cache()

				train_epoch_loss = train_running_loss / counter
				train_losses.append(train_epoch_loss)
				early_stopping(train_epoch_loss)
				if early_stopping.early_stop:
					break

			# Test the model
			model.eval()  # prep model for evaluation
			outputs = model(torch.FloatTensor(X_test).to(DEVICE))
			_, predicted = torch.max(outputs.data, 1)
			accus.append(accuracy_score(y_test, predicted.to('cpu').numpy()))
			f1s.append(f1_score(y_test, predicted.to('cpu').numpy(), average='macro'))

			model.to('cpu')

		return accus, f1s
