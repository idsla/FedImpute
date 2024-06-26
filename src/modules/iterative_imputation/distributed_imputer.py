from loguru import logger

from .utils import (
	initial_imputation, fit_one_feature, impute_one_feature, get_visit_indices, impute_one_feature2,
	get_clip_thresholds, get_estimator,
)
import numpy as np


class DistributedFeatureImputer:

	def __init__(
			self,
			missing_mask, estimator_num, estimator_cat, num_cols=None,
			initial_strategy_cat='most_frequent', initial_strategy_num='mean', clip=True, regression=False
	):
		# missing mask and missing info
		self.missing_mask = missing_mask
		self.missing_info = {}
		for col_idx in range(self.missing_mask.shape[1]):
			self.missing_info[col_idx] = self.get_missing_info(col_idx)

		# get estimators
		if estimator_num is None:
			estimator_num = 'huber'
		if estimator_cat is None:
			estimator_cat = 'logistic_regression_cv'
		# print(estimator_num, estimator_cat)
		self.estimator_num = get_estimator(estimator_num)
		self.estimator_cat = get_estimator(estimator_cat)

		# other parameters
		self.num_cols = num_cols
		self.initial_strategy_cat = initial_strategy_cat
		self.initial_strategy_num = initial_strategy_num
		self.clip = clip
		self.estimator_placeholder = None
		self.indicator_model = None
		self.regression = regression
		self.features_min = None
		self.features_max = None

	def initial_impute(self, X):
		# initialize
		num_cols = self.get_num_cols(X, self.num_cols)
		# imputation
		Xt = initial_imputation(X, self.initial_strategy_num, self.initial_strategy_cat, num_cols)
		return Xt

	def fit_feature(self, X, y, feature_idx):

		# initialize
		num_cols = self.get_num_cols(X, self.num_cols)

		# fit a model on to impute feature idx
		estimator = self.estimator_num if feature_idx < num_cols else self.estimator_cat
		estimator, losses, projection_matrix, ms_coef, lr = fit_one_feature(
			X, y, self.missing_mask, col_idx=feature_idx, estimator=estimator, num_cols=num_cols,
			regression=self.regression
		)

		# ms_coef = np.concatenate([lr.coef_[0], lr.intercept_])
		self.indicator_model = lr

		# save fitted estimator
		self.estimator_placeholder = estimator

		# get parameters and other information
		model_weights = np.concatenate([estimator.coef_, np.expand_dims(estimator.intercept_, 0)])

		# get missing info and losses
		missing_info = self.missing_info[feature_idx]

		return model_weights, losses, missing_info, projection_matrix, ms_coef

	def transform_feature(self, X, feature_idx, global_weights, update_weights=True):

		# initialize
		num_cols = self.get_num_cols(X, self.num_cols)

		# clip thresholds
		if self.clip:
			min_values = self.features_min
			max_values = self.features_max
		else:
			min_values = np.full((X.shape[1],), 0)
			max_values = np.full((X.shape[1],), 1)

		calibration = False
		if calibration:

			# impute X
			Xt = impute_one_feature2(
				X, self.missing_mask, feature_idx, self.estimator_placeholder, aggregation_weights=global_weights,
				min_value=min_values, max_value=max_values, num_cols=num_cols, indicator_model=self.indicator_model
			)
			return Xt
		else:
			# update parameters of local saved estimator
			if update_weights and global_weights is not None:
				self.estimator_placeholder.coef_ = global_weights[:-1]
				self.estimator_placeholder.intercept_ = global_weights[-1]

			# impute X
			Xt = impute_one_feature(
				X, self.missing_mask, feature_idx, self.estimator_placeholder,
				min_value=min_values, max_value=max_values, num_cols=num_cols
			)

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
