from .imputation_quality import rmse, ws_cols, sliced_ws
from .model_performance import get_evaluation_model, run_prediction_model


class Evaluator:

	def __init__(
			self,
			task_type, metrics, model, X_train, y_train, X_test, y_test, mask, seed,
			tune_params=None, scoring=None
	):
		self.task_type = task_type
		self.metrics = metrics
		self.model = model
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test
		self.mask = mask
		self.seed = seed
		self.tune_params = tune_params
		self.scoring = scoring

	def evaluation_imp(self, X_train_filled):
		ret = {}
		for metric_name in self.metrics:
			category, metric = metric_name.split('@')

			# imputation quality
			if category == 'imp':
				if metric == 'rmse':
					imp_score = rmse(X_train_filled, self.X_train, self.mask)
				elif metric == 'w2':
					imp_score = ws_cols(X_train_filled, self.X_train)
				elif metric == 'sliced_ws':
					imp_score = sliced_ws(X_train_filled, self.X_train)
				else:
					raise ValueError('Invalid metric name: {}'.format(metric_name))
				ret[metric_name] = imp_score

			# model performance
			elif category == 'model':
				evaluation_model, params_grid = get_evaluation_model(self.model, seed=self.seed)
				scores = run_prediction_model(
					evaluation_model, params_grid, self.task_type,
					X_train_filled, self.y_train, self.X_test, self.y_test, self.tune_params, self.scoring
				)
				ret.update(scores)

		return ret
