import numpy as np
import torch
from src.fed_imp.sub_modules.client.simple_client import SimpleClient
from typing import Dict, List
from src.utils import set_seed
from sklearn.metrics import f1_score, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegressionCV

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredServerCentralSklearn:

	def __init__(
			self,
			clients: Dict[int, SimpleClient], server_config: dict,
			pred_config: dict, test_data: np.ndarray = None, base_model='twonn'
	):

		self.clients = clients
		self.config = server_config
		self.seed = server_config['seed']
		self.base_model = base_model

		###########################################################################
		# Prediction model
		###########################################################################
		# model
		self.pred_model_params = pred_config.get(
			'model_params', {
				"num_hiddens": 64,
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

		# seeds
		self.rounds = server_config['pred_rounds']

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

		# centralized evaluation
		accus, f1s = [], []
		for s in seeds:
			set_seed(s)

			if self.base_model == 'mlp':
				clf = MLPClassifier(
					hidden_layer_sizes=(hidden_size, hidden_size),
					learning_rate_init=learning_rate,
					batch_size=batch_size,
					alpha=weight_decay,
					max_iter=train_epochs,
					early_stopping=True,
					validation_fraction=0.1,
					random_state=s
				)
			elif self.base_model == 'lr':
				clf = LogisticRegressionCV(
					max_iter=1000,
					random_state=s
				)
			else:
				raise ValueError(f'base model {self.base_model} not suppor')

			clf.fit(X_train, y_train)
			y_pred = clf.predict(X_test)
			accu = accuracy_score(y_test, y_pred)
			f1 = f1_score(y_test, y_pred, average='weighted')

			accus.append(accu)
			f1s.append(f1)

		return {
			"accu_mean": np.array(accus).mean(),
			"f1_mean": np.array(f1s).mean(),
			"accu_std": np.array(accus).std(),
			"f1_std": np.array(f1s).std()
		}

