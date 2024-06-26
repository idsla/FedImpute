import numpy as np
import torch
from loguru import logger
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, precision_score, balanced_accuracy_score, average_precision_score
from src.fed_imp.sub_modules.client.simple_client import SimpleClient
from typing import Dict, List
from src.utils import set_seed
from src.fed_imp.sub_modules.model.TwoNN import TwoNN
from src.fed_imp.sub_modules.model.logistic import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch.nn.functional as F
from sklearn.utils import class_weight

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PredServerCentralPytorch:

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

		self.pred_model_params['input_feature_dim'] = test_data.shape[1] - 1
		self.pred_model_params['output_classes_dim'] = len(np.unique(test_data[:, -1]))

		self.pred_training_params = pred_config.get(
			'train_params', {
				"batch_size": 300,
				"learning_rate": 0.001,
				"weight_decay": 0,
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
			seeds = list(range(42, 42 + self.rounds))

		validate_data, train_data = [], []
		for client_id, client in self.clients.items():
			validate_data.append(client.val_data)
			train_data.append(client.train_data)

		dataset = np.concatenate(train_data, axis=0)
		X_train, y_train = dataset[:, :-1], dataset[:, -1].astype(int)
		X_test, y_test = self.test_data[:, :-1], self.test_data[:, -1].astype(int)
		validate_dataset = np.concatenate(validate_data, axis=0)
		X_validate, y_validate = validate_dataset[:, :-1], validate_dataset[:, -1].astype(int)

		# label_unique, counts = np.unique(y_train, return_counts=True)
		# class_weights = [(sum(counts)/c) for c in counts]
		# example_weights = [class_weights[y] for y in y_train]
		# sampler = WeightedRandomSampler(example_weights, len(example_weights), replacement=True)
		# class_weights = class_weight.compute_class_weight(
		# 	class_weight='balanced', classes = np.unique(y_train), y = y_train)
		#class_weights = torch.tensor(class_weights,dtype=torch.float)

		################################################################################################
		# Training prediction model
		################################################################################################
		hidden_size = self.pred_model_params['num_hiddens']
		train_epochs = self.pred_training_params['pred_round']
		batch_size = self.pred_training_params['batch_size']
		learning_rate = self.pred_training_params['learning_rate']
		weight_decay = self.pred_training_params['weight_decay']

		# centralized evaluation
		accus, f1s, roc_aucs, prc_aucs = [], [], [], []
		for s in seeds:
			set_seed(s)
			train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
			train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
			# train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=False)
			# early_stopping = EarlyStopping()

			# Model. loss and optimizer
			if self.base_model == 'twonn':
				model = TwoNN(
					in_features=X_train.shape[1],
					num_classes=len(np.unique(y_train)),
					num_hiddens=hidden_size
				)
			elif self.base_model == 'lr':
				model = LogisticRegression(
					in_features=X_train.shape[1],
					num_classes=len(np.unique(y_train))
				)
			else:
				raise ValueError('base model not found')

			#criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(DEVICE),reduction='mean')
			criterion = torch.nn.CrossEntropyLoss(reduction='mean')
			optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

			# Train the model
			train_losses, val_losses = [], []
			best_accus, best_f1s, best_rocs, best_prcs = [], [], [], []
			for epoch in range(train_epochs):
				model.to(DEVICE)
				model.train()  # prep model for training

				train_running_loss = 0.0
				counter = 0
				for i, (X_train_, y_train_) in enumerate(train_dataloader):
					counter += 1
					# Move tensors to the configured device
					X_train_, y_train_ = X_train_.to(DEVICE), y_train_.to(DEVICE)

					# Forward pass
					outputs = model(X_train_)
					loss = criterion(outputs, y_train_)

					# Backward and optimize
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					train_running_loss += loss.item()

					if DEVICE == 'cuda':
						torch.cuda.empty_cache()

				train_epoch_loss = train_running_loss / counter
				train_losses.append(train_epoch_loss)
				# early_stopping(train_epoch_loss)

				model.eval()  # prep model for evaluation
				outputs = model(torch.FloatTensor(X_test).to(DEVICE))
				_, predicted = torch.max(outputs.data, 1)
				probabilities = F.softmax(outputs, dim=1)
				accu = accuracy_score(y_test, predicted.to('cpu').numpy())
				if len(np.unique(y_test)) > 2:	
					f1 = f1_score(y_test, predicted.to('cpu').numpy(), zero_division=0, average='micro')
				else:
					f1 = f1_score(y_test, predicted.to('cpu').numpy(), zero_division=0)

				if probabilities.shape[1] == 2:
					roc_auc = roc_auc_score(
						y_test, probabilities.detach().to('cpu').numpy()[:, 1]
					)
					prc_auc = average_precision_score(
						y_test, probabilities.detach().to('cpu').numpy()[:, 1]
					)
				else:
					roc_auc = roc_auc_score(
						y_test, probabilities.detach().to('cpu').numpy(), multi_class='ovr', average='micro'
					)
					prc_auc = average_precision_score(
						y_test, probabilities.detach().to('cpu').numpy(), average='micro'
					)
				
				
				best_accus.append(accu)
				best_f1s.append(f1)
				best_rocs.append(roc_auc)
				best_prcs.append(prc_auc)
				# recall = recall_score(y_test, predicted.to('cpu').numpy(), zero_division=0)
				# precision = precision_score(y_test, predicted.to('cpu').numpy(), zero_division=0)
				# balanced_accu = balanced_accuracy_score(y_test, predicted.to('cpu').numpy())

				outputs = model(torch.FloatTensor(X_validate).to(DEVICE))
				_, predicted = torch.max(outputs.data, 1)
				val_accu = accuracy_score(y_validate, predicted.to('cpu').numpy())
				val_f1 = f1_score(y_validate, predicted.to('cpu').numpy(), average='macro')
				

				if epoch % 100 == 0:
					logger.info(
						'Round: {}, test_accu: {:.4f}, test_f1: {:.4f} test_auroc: {:.4f} test_auprc: {:.4f} train_loss: {:.4f} '
						'val_accu: '
						'{:.4f} val_f1: {:.4f}'.format(
							epoch, accu, f1, roc_auc, prc_auc, train_epoch_loss, val_accu, val_f1, 
						)
					)

			# if early_stopping.early_stop:
			# 	break

			# Test the model
			accus.append(np.array(best_accus).max())
			f1s.append(np.array(best_f1s).max())
			roc_aucs.append(np.array(best_rocs).max())
			prc_aucs.append(np.array(best_prcs).max())

			model.to('cpu')

		return {
			"accu_mean": np.array(accus).mean(),
			"f1_mean": np.array(f1s).mean(),
			"roc_auc_mean": np.array(roc_aucs).mean(),
			"prc_auc_mean": np.array(prc_aucs).mean(),
			"accu_std": np.array(accus).std(),
			"f1_std": np.array(f1s).std(),
			"roc_auc_std": np.array(roc_aucs).std(),
			"prc_auc_std": np.array(prc_aucs).std()
		}
