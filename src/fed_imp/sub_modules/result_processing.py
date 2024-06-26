from functools import reduce
from typing import List
from matplotlib import pyplot as plt
from src.modules.utils import plot_lines
import base64
from io import BytesIO
import numpy as np


def average_n_rounds_result(client_rets):
	avg_client_ret = {}
	metrics = list(client_rets[0].keys())
	for metric in metrics:
		metric_avg_obj = {}
		# collect results for each client for this metric
		for client_ret in client_rets:
			metric_obj = client_ret[metric]
			for k, v in metric_obj.items():
				if k not in metric_avg_obj:
					metric_avg_obj[k] = [v]
				else:
					metric_avg_obj[k].append(v)
		# average results for each client for this metric
		for k, v in metric_avg_obj.items():
			metric_avg_obj[k] = list(np.array(v).mean(axis=0))

		# save result
		avg_client_ret[metric] = metric_avg_obj

	return avg_client_ret


def processing_imputation_result(client_imp_history: List):

	# helper function
	def combine_dicts(d1, d2, metric_field):
		for client, client_item in d2.items():
			if client not in d1:
				d1[client] = [client_item[metric_field]]
			else:
				d1[client].append(client_item[metric_field])
		return d1

	# clear local epoch result
	client_history = [item for item in client_imp_history if item[0] == 'server']
	x = [item[1] for item in client_history]

	# client avg result
	clients_ids = list(client_history[0][2]["metrics"].keys())
	metrics = list(client_history[0][2]['metrics'][clients_ids[0]].keys())
	clients_ret = {}  # {metric1: {client1: [metric], client2: [metric], client_avg: [metric]}}
	for metric in metrics:
		# avg metric
		avg_metric = [sum(item[metric] for item in round_ret["metrics"].values()) / len(round_ret["metrics"])
		              for _, _, round_ret in client_history]
		# all clients metric
		clients_metric = reduce(
			lambda d1, d2: combine_dicts(d1, d2, metric), [item["metrics"] for _, _, item in client_history], {}
		)
		clients_ret[metric] = clients_metric
		clients_ret[metric]['client_avg'] = avg_metric

	return clients_ret, x


def visualizing_clients_result(clients_ret, metrics=None, x=None):
	if metrics is None:
		metrics = ['imp@rmse', 'imp@w2']

	n_cols = 4 if len(metrics) > 4 else len(metrics)
	n_rows = len(metrics) // 4 + 1

	fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(n_cols * 5, n_rows * 5))
	for metric_idx, metric in enumerate(metrics):
		client_ret_metric = clients_ret[metric]
		plot_lines(
			client_ret_metric, emphasize_lines=['client_avg'], name=metric, x=x,
			axes=axes[metric_idx // n_cols][metric_idx % n_cols]
			)

	plt.tight_layout()
	tmp_file = BytesIO()
	fig.savefig(tmp_file, format='png')
	#plt.show()
	plt.close(fig)
	encoded = base64.b64encode(tmp_file.getvalue()).decode('utf-8')
	return encoded
