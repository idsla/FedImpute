import numpy as np
from sklearn.linear_model import Lasso
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import (
	BayesianRidge, LinearRegression, RidgeCV, LassoCV, LogisticRegressionCV,
	LogisticRegression, Ridge, TheilSenRegressor, HuberRegressor,
)
from sklearn.metrics import mean_squared_error, r2_score

def initial_imputation(X, initial_strategy_num, initial_strategy_cat, num_cols):
	X_copy = X.copy()
	# initial imputation for numerical columns
	X_num = X_copy[:, :num_cols]
	if initial_strategy_num == 'mean':
		simple_imp = SimpleImputer(strategy='mean')
		X_num_t = simple_imp.fit_transform(X_num)
	elif initial_strategy_num == 'median':
		simple_imp = SimpleImputer(strategy='median')
		X_num_t = simple_imp.fit_transform(X_num)
	elif initial_strategy_num == 'zero':
		simple_imp = SimpleImputer(strategy='constant', fill_value=0)
		X_num_t = simple_imp.fit_transform(X_num)
	else:
		raise ValueError("initial_strategy_num must be one of 'mean', 'median', 'zero'")

	if num_cols == X.shape[1]:
		return X_num_t

	# initial imputation for categorical columns
	X_cat = X_copy[:, num_cols:]
	if initial_strategy_cat == 'mode':
		simple_imp = SimpleImputer(strategy='most_frequent')
		X_cat_t = simple_imp.fit_transform(X_cat)
	elif initial_strategy_cat == 'other':
		simple_imp = SimpleImputer(strategy='constant', fill_value=-1)
		X_cat_t = simple_imp.fit_transform(X_cat)
	else:
		raise ValueError("initial_strategy_cat must be one of 'mode', 'other'")

	Xt = np.concatenate((X_num_t, X_cat_t), axis=1)
	return Xt


def fit_one_feature(X_filled, y, missing_mask, col_idx, estimator, num_cols, compute_proj=False, regression = False):

	# calculate correct num_cols (col_idx is numerical column then after remove it, num_cols will decrease by 1)
	# if col_idx < num_cols:
	# 	num_cols = num_cols - 1

	# give observed part of col_idx to estimator
	row_mask = missing_mask[:, col_idx]
	X_train = X_filled[~row_mask][:, np.arange(X_filled.shape[1]) != col_idx]
	y_train = X_filled[~row_mask][:, col_idx]

	#one hot encoding for categorical columns
	# X_train_cat = X_train[:, num_cols:]
	# if X_train_cat.shape[1] > 0:
	# 	onehot_encoder = OneHotEncoder(max_categories=5, drop="if_binary")
	# 	X_train_cat = onehot_encoder.fit_transform(X_train_cat)
	# 	X_train = np.concatenate((X_train[:, :num_cols], X_train_cat), axis=1)
	# else:
	# 	X_train = X_train[:, :num_cols]
	# fit estimator
	estimator.fit(X_train, y_train)

	# compute losses
	y_pred = estimator.predict(X_train)
	r2 = r2_score(y_train, y_pred)  # R2 score
	rmse = np.sqrt(mean_squared_error(y_train, y_pred))  # RMSE

	# projection matrix
	if compute_proj:
		X_train_ext = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis=1)
		XtX_pinv = np.linalg.pinv(X_train_ext @ X_train_ext.T + 1e5 * np.identity(X_train_ext.shape[0]))
		projection_matrix = X_train_ext.T @ XtX_pinv @ X_train_ext
	else:
		projection_matrix = None

	# missing indicator prediction model
	if not regression:
		oh = OneHotEncoder(drop='first')
		y_oh = oh.fit_transform(y.reshape(-1, 1)).toarray()
		#X_ = np.concatenate([X_filled[:, np.arange(X_filled.shape[1]) != col_idx]], axis=1)
		X_ = np.concatenate([X_filled], axis=1)
	else:
		X_ = np.concatenate([X_filled], axis=1)
	y_ = row_mask
	if row_mask.sum() == 0:
		coef = np.zeros(X_.shape[1]) + 0.001
		lr = None
	else:
		lr = LogisticRegressionCV(
			Cs=[1e-1], cv=StratifiedKFold(3), random_state=0, max_iter=1000, n_jobs=-1, class_weight='balanced'
		)
		lr.fit(X_, y_)
		coef = np.concatenate([lr.coef_[0], lr.intercept_])

	return estimator, {"r2": r2, "rmse": rmse}, projection_matrix, coef, lr


def impute_one_feature(X_filled, missing_mask, col_idx, estimator, num_cols, min_value=None, max_value=None):

	row_mask = missing_mask[:, col_idx]

	# calculate correct num_cols (col_idx is numerical column then after remove it, num_cols will decrease by 1)
	# if col_idx < num_cols:
	# 	num_cols = num_cols - 1

	# if no missing values, dont predict
	if np.sum(row_mask) == 0:
		return X_filled

	# predict missing values
	X_test = X_filled[row_mask][:, np.arange(X_filled.shape[1]) != col_idx]

	# one hot encoding for categorical columns
	# X_test_cat = X_test[:, num_cols:]
	# if X_test_cat.shape[1] > 0:
	# 	onehot_encoder = OneHotEncoder(sparse=False, max_categories=10, drop="if_binary")
	# 	X_test_cat = onehot_encoder.fit_transform(X_test_cat)
	# 	X_test = np.concatenate((X_test[:, :num_cols], X_test_cat), axis=1)
	# else:
	# 	X_test = X_test[:, :num_cols]

	# impute missing data
	imputed_values = estimator.predict(X_test)
	imputed_values = np.clip(imputed_values, min_value[col_idx], max_value[col_idx])
	X_filled[row_mask, col_idx] = np.squeeze(imputed_values)

	return X_filled


def impute_one_feature2(
		X_filled, missing_mask, col_idx, estimator, num_cols, aggregation_weights, indicator_model,
		min_value=None, max_value=None
):

	row_mask = missing_mask[:, col_idx]

	# calculate correct num_cols (col_idx is numerical column then after remove it, num_cols will decrease by 1)
	if col_idx < num_cols:
		num_cols = num_cols - 1

	# if no missing values, don‘t predict
	if np.sum(row_mask) == 0:
		return X_filled

	# predict missing values
	X_test = X_filled[row_mask][:, np.arange(X_filled.shape[1]) != col_idx]

	# one hot encoding for categorical columns
	X_test_cat = X_test[:, num_cols:]
	if X_test_cat.shape[1] > 0:
		onehot_encoder = OneHotEncoder(sparse=False, max_categories=10, drop="if_binary")
		X_test_cat = onehot_encoder.fit_transform(X_test_cat)
		X_test = np.concatenate((X_test[:, :num_cols], X_test_cat), axis=1)
	else:
		X_test = X_test[:, :num_cols]

	# impute missing data using local estimator
	imputed_values1 = estimator.predict(X_test)
	# impute missing data using global estimator
	estimator.coef_ = aggregation_weights[:-1]
	estimator.intercept_ = aggregation_weights[-1]
	imputed_values2 = estimator.predict(X_test)

	# combine local and global predictions
	missing_ratio = indicator_model.predict_proba(X_test)[:, 1] + 0.0001
	# print(missing_ratio)
	# print(np.linalg.norm(imputed_values1), np.linalg.norm(imputed_values2))
	imputed_values = (imputed_values2 - imputed_values1 * (1 - missing_ratio)) / missing_ratio
	# print(imputed_values)
	imputed_values = np.clip(imputed_values, min_value[col_idx], max_value[col_idx])
	X_filled[row_mask, col_idx] = np.squeeze(imputed_values)

	return X_filled


########################################################################################################################
# Helper functions
########################################################################################################################
def get_visit_indices(visit_sequence, missing_mask):
	frac_of_missing_values = missing_mask.mean(axis=0)
	missing_values_idx = np.flatnonzero(frac_of_missing_values)

	if visit_sequence == 'roman':
		ordered_idx = missing_values_idx
	elif visit_sequence == 'arabic':
		ordered_idx = missing_values_idx[::-1]
	elif visit_sequence == 'random':
		ordered_idx = missing_values_idx.copy()
		np.random.shuffle(ordered_idx)
	elif visit_sequence == 'ascending':
		n = len(frac_of_missing_values) - len(missing_values_idx)
		ordered_idx = np.argsort(frac_of_missing_values, kind="mergesort")[n:]
	elif visit_sequence == 'descending':
		n = len(frac_of_missing_values) - len(missing_values_idx)
		ordered_idx = np.argsort(frac_of_missing_values, kind="mergesort")[n:][::-1]
	else:
		raise ValueError("Invalid choice for visit order: %s" % visit_sequence)

	return ordered_idx


def get_clip_thresholds(X, clip=False):
	if clip:
		min_values = X.min(axis=0)
		max_values = X.max(axis=0)
	else:
		min_values = np.full((X.shape[1],), -np.inf)
		max_values = np.full((X.shape[1],), np.inf)

	return min_values, max_values


def check_convergence(x, y, threshold):
	return np.linalg.norm(x - y) < threshold


def get_estimator(estimator_name):
	# TODO: ADD SEEDED RANDOM STATE
	if estimator_name == 'bayesian_ridge':
		return BayesianRidge()
	elif estimator_name == 'linear_regression':
		return LinearRegression(n_jobs=-1)
	elif estimator_name == 'ridge':
		return Ridge(alpha=1.0, random_state=0)
	elif estimator_name == 'lasso':
		return Lasso(alpha=0.1, random_state=0)
	elif estimator_name == 'theilsen':
		return TheilSenRegressor(random_state=0, n_jobs=-1)
	elif estimator_name == 'huber':
		return HuberRegressor()
	elif estimator_name == 'ridge_cv':
		return RidgeCV(alphas=[0.1, 1.0, 10.0])
	elif estimator_name == 'lasso_cv':
		return LassoCV(alphas=[0.1, 1.0, 10.0])
	elif estimator_name == 'logistic':
		return LogisticRegression(penalty='l1', n_jobs=-1)
	elif estimator_name == 'logistic_cv':
		return LogisticRegressionCV(Cs=[0.1, 1.0, 10.0], penalty='l1', solver='saga')
	elif estimator_name == 'mlp':
		return MLPRegressor(hidden_layer_sizes=(16, 16), max_iter=1000, random_state=0)
	else:
		raise ValueError('Unknown estimator name: {}'.format(estimator_name))
