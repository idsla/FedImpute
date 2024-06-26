import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.modules.iterative_imputation.distributed_imputer import DistributedFeatureImputer
from src.modules.evaluation.evaluation import Evaluator
from sklearn.linear_model import LogisticRegression
import numpy as np
from loguru import logger
from src.fed_imp.sub_modules.dataloader import construct_tensor_dataset
import pandas as pd
import matplotlib.pyplot as plt
import missingno

class Client:
	"""
	Client class for federated learning
	"""

	def __init__(
			self,
			client_id,
			client_data, client_data_config: dict,
			imputation_config: dict,
			debug: bool = False,
			seed: int = 21
	):
		################################################################################################################
		# Data
		################################################################################################################
		self.X_train_filled = None
		self.X_train_filled_pt = None
		self.tol = 1e-3
		self.client_id = client_id

		self.X_train, self.y_train = client_data.get('train_data')[:, :-1], client_data.get('train_data')[:, -1]
		self.X_test, self.y_test = client_data.get('test_data')[:, :-1], client_data.get('test_data')[:, -1]
		self.X_train_ms = client_data.get('train_data_ms')[:, :-1]
		self.client_data_config = client_data_config
		self.seed = seed
		self.convergency = False
		self.features_min: np.ndarray = self.X_train.min(axis=0)
		self.features_max: np.ndarray = self.X_train.max(axis=0)

		# get original missing mask 0-1 matrix
		self.missing_mask = np.isnan(self.X_train_ms).astype(bool)
		self.num_cols = self.client_data_config.get('num_cols', self.X_train.shape[1])
		self.task_type = self.client_data_config.get('task_type')

		# debug
		self.debug = debug
		# missingno.matrix(pd.DataFrame(self.X_train_ms))
		# plt.show()
		self.top_k_idx = None
		if self.task_type == 'classification':
			self.regression = False
		else:
			self.regression = True
		################################################################################################################
		# imputation parameters
		################################################################################################################
		self.initial_strategy_cat = imputation_config.get('initial_strategy_cat', 'most_frequent')
		self.initial_strategy_num = imputation_config.get('initial_strategy_num', 'mean')
		self.imp_estimator_num = imputation_config.get('imp_estimator_num', 'ridge')
		self.imp_estimator_cat = imputation_config.get('imp_estimator_cat', 'logistic_cv')
		self.imp_clip = imputation_config.get('clip', False)

		self.imputation_model = DistributedFeatureImputer(
			missing_mask=self.missing_mask, estimator_num=self.imp_estimator_num, estimator_cat=self.imp_estimator_cat,
			initial_strategy_cat=self.initial_strategy_cat, initial_strategy_num=self.initial_strategy_num,
			clip=self.imp_clip, num_cols=self.num_cols, regression=self.regression
		)

		################################################################################################################
		# evaluation of imputation
		################################################################################################################
		if self.task_type == 'classification':
			self.imp_evaluation_model = imputation_config.get('imp_evaluation_model', 'logistic')
			self.imp_evaluation_params = imputation_config.get('imp_evaluation_params', {})
			self.evaluation_metrics = ['imp@rmse', 'imp@w2', 'imp@sliced_ws']
		else:
			self.imp_evaluation_model = imputation_config.get('imp_evaluation_model', 'ridge')
			self.evaluation_metrics = ['imp@rmse', 'imp@w2', 'imp@sliced_ws']
			self.imp_evaluation_params = imputation_config.get('imp_evaluation_params', {})

		self.evaluator = Evaluator(
			task_type=self.task_type, metrics=self.evaluation_metrics, model=self.imp_evaluation_model,
			X_train=self.X_train, y_train=self.y_train, X_test=self.X_test, y_test=self.y_test,
			mask=self.missing_mask, seed=seed, tune_params=self.imp_evaluation_params['tune_params']
		)

		################################################################################################################
		# prediction model
		################################################################################################################
		self.pred_model = None  # server will copy its model to clients by using this hook
		self.train_data = None
		self.val_data = None
		self.local_pred_dataset = None
		self.local_pred_dataset_val = None
		self.train_dataloader = None
		self.val_data_loader = None

	def initialize(self, feature_indices):
		return self.imputation_model_grad.make_data_loader_and_model(self.X_train_filled, feature_indices)

	def fit(self, fit_task: str, fit_instruction: dict):
		if fit_task == 'fit_imputation_model':
			model_weights, losses, missing_info, projection_matrix, ms_coef = self.imputation_model.fit_feature(
				self.X_train_filled, self.y_train, feature_idx=fit_instruction.get('feature_idx')
			)
			return model_weights, losses, missing_info, projection_matrix, ms_coef
		elif fit_task == 'fit_imputation_model_grad':
			model_weights, losses, missing_info = self.imputation_model_grad.fit_feature(
				feature_idx=fit_instruction.get('feature_idx'), optimizer=fit_instruction.get('optimizer'),
				local_epochs=fit_instruction.get('local_epochs'), learning_rate=fit_instruction.get('lr'),
				global_weights=fit_instruction.get('global_weights')
			)
			return model_weights, losses, missing_info, None, None
		elif fit_task == 'fit_prediction_model':
			pass
			# self.prediction_model.fit(self.X_train_filled, self.y_train)
			# model_weights = (self.prediction_model.coef_, self.prediction_model.intercept_)
			# return model_weights

	def transform(self, transform_task: str, transform_instruction: dict, global_weights, update_weights: bool = True):
		if transform_task == 'impute_data':
			if not self.convergency:
				self.X_train_filled = self.imputation_model.transform_feature(
					X=self.X_train_filled, feature_idx=transform_instruction.get('feature_idx'),
					global_weights=global_weights, update_weights=update_weights
				)
		else:
			raise NotImplementedError

	def evaluate(
			self, evaluate_task: str, evaluate_instruction: dict, global_weights=None, update_weights: bool = True
	):
		if evaluate_task == 'evaluate_imputation_model':
			ret = self.evaluator.evaluation_imp(self.X_train_filled)
			return ret
		elif evaluate_task == 'evaluate_prediction_model':
			return {}
		return {}

	####################################################################################################################
	# helper functions
	####################################################################################################################
	def get_initial_values(self):
		ret_mean, ret_max, ret_min, sample_size, ms_info = [], [], [], [], []
		for idx in range(self.X_train_ms.shape[1]):
			ret_mean.append(np.nanmean(self.X_train_ms[:, idx]))
			ret_max.append(np.nanmax(self.X_train_ms[:, idx]))
			ret_min.append(np.nanmin(self.X_train_ms[:, idx]))
			ms_info.append(self.imputation_model.get_missing_info(idx)['missing_cell_pct'])
			sample_size.append(self.imputation_model.get_missing_info(idx)['sample_row_pct'])
		return np.array(ret_mean), np.array(ret_max), np.array(ret_min), np.array(sample_size), np.array(ms_info)

	def initial_impute(self, aggregated_values):
		if aggregated_values is None:
			self.X_train_filled = self.imputation_model.initial_impute(self.X_train_ms)
			self.X_train_filled_pt = self.X_train_filled.copy()
		else:
			self.X_train_filled = self.X_train_ms.copy()
			for idx in range(self.X_train_ms.shape[1]):
				indices = np.where(np.isnan(self.X_train_ms[:, idx]))[0]
				self.X_train_filled[indices, idx] = aggregated_values[idx]

			self.X_train_filled_pt = self.X_train_filled.copy()

	def check_convergency(self):
		if self.convergency is True:
			return True
		else:
			# check convergency of the data matrix
			inf_norm = np.linalg.norm(self.X_train_filled - self.X_train_filled_pt, ord=np.inf, axis=None)
			normalized_tol = self.tol * np.max(np.abs(self.X_train_filled[~self.missing_mask]))
			#print(self.client_id, inf_norm, normalized_tol)
			if inf_norm < normalized_tol:
				self.convergency = True

			self.X_train_filled_pt = self.X_train_filled.copy()

		return self.convergency

	####################################################################################################################
	# prediction model helper functions
	####################################################################################################################
	def local_train_pred(self, local_epoch, lr, weight_decay):

		pred_model = self.pred_model

		# pred model1
		torch.manual_seed(self.seed)
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		pred_model.train()
		pred_model.to(device)

		optimizer = torch.optim.Adam(pred_model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=True)
		#compiled_model = torch.compile(pred_model)

		for e in range(local_epoch):
			for data, labels in self.train_dataloader:  # TODO
				data, labels = data.float().to(device), labels.long().to(device)

				optimizer.zero_grad()
				outputs = pred_model(data)
				loss = torch.nn.CrossEntropyLoss()(outputs, labels)

				loss.backward()
				optimizer.step()

				if device == "cuda":
					torch.cuda.empty_cache()

		pred_model.to("cpu")

	def get_sample_size(self):
		return self.X_train_filled.shape[0]

	def local_dataset(self):
		X_train_filled, X_val_filled, y_train, y_val = train_test_split(
			self.X_train_filled, self.y_train, test_size=0.1, random_state=self.seed, stratify=self.y_train
		)

		self.train_data = np.concatenate((X_train_filled, y_train.reshape(-1, 1)), axis=1)
		self.val_data = np.concatenate((X_val_filled, y_val.reshape(-1, 1)), axis=1)

	def pred_data_setup(self, batch_size):

		self.local_pred_dataset = construct_tensor_dataset(self.train_data[:, :-1], self.train_data[:, -1])
		self.local_pred_dataset_val = construct_tensor_dataset(self.val_data[:, :-1], self.val_data[:, -1])
		self.train_dataloader = DataLoader(self.local_pred_dataset, batch_size=batch_size, shuffle=True)
		self.val_data_loader = DataLoader(self.local_pred_dataset_val, batch_size=batch_size, shuffle=False)

	def update_imp_model_minmax(self):
		self.imputation_model.features_max = self.features_max
		self.imputation_model.features_min = self.features_min

