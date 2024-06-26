import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Ridge, RidgeCV
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.metrics import (
	accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
	r2_score, mean_squared_error, mean_absolute_error,
)
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


def run_prediction_model(
		model, param_grids, task_type, X_train, y_train, X_test, y_test,
		tune_params=None, scoring=None, seed=21
):
	if tune_params == 'notune' or param_grids is None:
		model.fit(X_train, y_train)
		scores = pred_eval(model, X_test, y_test, task_type)
		return scores
	else:
		# get scoring information
		if scoring is None:
			if task_type == 'classification':
				scoring = 'accuracy'
			else:
				scoring = 'neg_mean_squared_error'

		# grid search
		if tune_params == 'gridsearch':
			grid_search = GridSearchCV(
				model,
				param_grids,
				scoring=scoring,
				n_jobs=-1,
				verbose=0,
				cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
			)
			grid_search.fit(X_train, y_train)
			estimator = grid_search.best_estimator_
			estimator.fit(X_train, y_train)
			scores = pred_eval(estimator, X_test, y_test, task_type)
			return scores
		elif tune_params in ['random', 'bayesian', "optuna"]:
			raise ValueError("not implement")

			# tune_search = TuneSearchCV(
			# 	model,
			# 	param_grids,
			# 	scoring=scoring,
			# 	n_jobs=-1,
			# 	verbose=0,
			# 	random_state=seed,
			# 	early_stopping=True,
			# 	max_iters=10,
			# 	n_trials=10,
			# 	search_optimization=tune_params,
			# )
			# tune_search.fit(X_train, y_train)
			# scores = pred_eval(tune_search, X_test, y_test, task_type)
			# return scores
		else:
			raise ValueError('Invalid tuning method: {}'.format(tune_params))


def pred_eval(model, X_test, y_test, task_type):
	y_pred = model.predict(X_test)
	if hasattr(model, 'predict_proba'):
		y_pred_proba = model.predict_proba(X_test)
	else:
		y_pred_proba = None
	if task_type == 'classification':
		scores = clf_performance(y_test, y_pred, y_pred_proba)
	elif task_type == 'regression':
		scores = reg_performance(y_test, y_pred)
	else:
		raise ValueError('Invalid task type: {}'.format(task_type))
	return scores


def clf_performance(y_true, y_pred, y_pred_proba=None):
	if y_pred_proba is not None:
		# if len(np.unique(y_true)) != y_pred_proba.shape[1]:
		# 	diff = len(np.unique(y_true)) - y_pred_proba.shape[1]
		# 	classes =
		# 	if diff > 0:
		# 		y_pred_proba = np.concatenate([y_pred_proba, np.zeros((y_pred_proba.shape[0], diff))], axis=1)
		# 	else:
		# 		y_pred_proba = y_pred_proba[:, :len(np.unique(y_true))]
		if y_pred_proba.shape[1] == 2:
			roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
		else:
			roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
	else:
		roc_auc = 0
	return {
		'accuracy': accuracy_score(y_true, y_pred),
		'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
		'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
		'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
		'roc_auc': roc_auc
	}


def reg_performance(y_true, y_pred):
	return {
		'r2': r2_score(y_true, y_pred),
		'mse': mean_squared_error(y_true, y_pred),
		'mae': mean_absolute_error(y_true, y_pred)
	}


def get_evaluation_model(model_name, seed=21, scoring=None):
	####################################################################################################################
	# Classification models
	####################################################################################################################
	# logistic regression
	if model_name == 'logistic':
		model = LogisticRegression(random_state=seed, max_iter=1000, n_jobs=-1)
		param_grids = {
			'penalty': ['l2'],
			'C': [0.01, 0.1, 1.0, 5, 10.0],
		}
	# logistic regression with cross validation
	elif model_name == 'logistic_cv':
		if scoring is None:
			scoring = 'accuracy'
		model = LogisticRegressionCV(random_state=seed, cv=3, max_iter=1000, n_jobs=-1, scoring=scoring)
		param_grids = None
	# regression forest classifier
	elif model_name == 'rf_clf':
		model = RandomForestClassifier(random_state=seed, n_jobs=-1)
		param_grids = {
			'n_estimators': [10, 100, 1000],
			'max_depth': [3, 5, 10],
			'max_features': ['sqrt', 'log2'],
		}
	# linear support vector classifier
	elif model_name == 'linear_svc':
		model = LinearSVC(random_state=seed, dual=False)
		param_grids = {
			'C': [0.1, 0.5, 1.0, 5, 10.0, 20],
		}
	# light gradient boosting machine classifier
	elif model_name == 'lgbm_clf':
		model = LGBMClassifier(random_state=seed)
		param_grids = {
			'learning_rate': [0.01, 0.05, 0.1],
			'n_estimators': [10, 100, 1000],
			'max_depth': [3, 5, 10],
			'num_leaves': [50, 500, 2000],
			'lambda_l1': [0.0, 0.1, 1.0]
		}
	####################################################################################################################
	# Regression models
	####################################################################################################################
	# ridge regression
	elif model_name == 'ridge':
		model = Ridge(random_state=seed)
		param_grids = {
			'alpha': [0.1, 1.0, 10.0],
		}
	# ridge regression with cross validation
	elif model_name == 'ridge_cv':
		if scoring is None:
			scoring = 'neg_mean_squared_error'
		model = RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3, scoring=scoring)
		param_grids = None
	# regression forest regressor
	elif model_name == 'rf_reg':
		model = RandomForestRegressor(random_state=seed, n_estimators=300, n_jobs=-1, max_features='sqrt')
		param_grids = {
			'n_estimators': [10, 100, 1000],
			'max_depth': [3, 5, 10],
			'max_features': ['auto', 'sqrt', 'log2'],
		}
	# linear support vector regressor
	elif model_name == 'linear_svr':
		model = LinearSVR(random_state=seed)
		param_grids = {
			'penalty': ['l1', 'l2'],
			'C': [0.1, 1.0, 10.0],
			'epsilon': [0.1, 0.5, 1.0],
			'max_iter': [100, 1000, 3000],
		}
	# light gradient boosting machine regressor
	elif model_name == 'lgbm_reg':
		model = LGBMRegressor(random_state=seed)
		param_grids = {
			'learning_rate': [0.01, 0.05, 0.1],
			'n_estimators': [10, 100, 1000],
			'max_depth': [3, 5, 10],
			'num_leaves': [50, 500, 2000],
			'lambda_l1': [0.0, 0.1, 1.0]
		}
	else:
		raise ValueError(f'Unknown model name: {model_name}')

	return model, param_grids
