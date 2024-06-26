from .utils import (
	initial_imputation, fit_one_feature, impute_one_feature, get_visit_indices, get_clip_thresholds,
	get_estimator
)
import numpy as np


class LocalIterativeImputer:

	def __init__(
			self,
			estimator_num = None, estimator_cat = None, max_iter=20,
			initial_strategy_cat='mode', initial_strategy_num='mean',
			impute_order='ascending', impute_mode='instant', clip=False, seed=221
	):
		# get estimators
		if estimator_num is None:
			estimator_num = 'ridge_cv'
		if estimator_cat is None:
			estimator_cat = 'logistic_cv'

		self.estimator_num = get_estimator(estimator_num)
		self.estimator_cat = get_estimator(estimator_cat)

		self.estimator_cat = estimator_cat
		self.estimators = None  # list of estimators for each feature
		self.max_iter = max_iter
		self.initial_strategy_cat = initial_strategy_cat
		self.initial_strategy_num = initial_strategy_num
		self.impute_order = impute_order
		self.impute_mode = impute_mode
		self.clip = clip
		self.seed = seed

	def fit_transform(self, X, y=None, num_cols=None):
		if num_cols is None:
			num_cols = X.shape[1]

		min_values, max_values = get_clip_thresholds(X, self.clip)
		np.random.seed(self.seed)

		# missing mask
		missing_mask = np.isnan(X)

		# initial imputation
		X_filled = initial_imputation(X, self.initial_strategy_num, self.initial_strategy_cat, num_cols)

		# get imputed indices
		impute_indices = get_visit_indices(self.impute_order, missing_mask)

		# impute by impute indices
		for _ in range(self.max_iter):
			if self.impute_mode == 'instant':
				# fit and impute one feature at a time
				for col_idx in impute_indices:
					estimator = self.estimator_num if col_idx < num_cols else self.estimator_cat
					estimator,_ = fit_one_feature(X_filled, y, missing_mask, col_idx, estimator, num_cols)
					X_filled = \
						impute_one_feature(X_filled, missing_mask, col_idx, estimator, num_cols, min_values, max_values)
			else:
				# fit all features, then impute
				estimators = []
				for col_idx in impute_indices:
					estimator = self.estimator_num if col_idx < num_cols else self.estimator_cat
					estimator,_ = fit_one_feature(X_filled, y, missing_mask, col_idx, estimator, num_cols)
					estimators.append(estimator)
				for idx, col_idx in enumerate(impute_indices):
					estimator = estimators[idx]
					X_filled = \
						impute_one_feature(X_filled, missing_mask, col_idx, estimator, min_values, max_values)

		return X_filled


