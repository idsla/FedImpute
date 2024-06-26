import numpy as np
from config import settings


def fedwavgcl(weights, losses, missing_infos, client_groups):
	'''Two Levels Clustering'''

	weights = np.array(list(weights.values()))

	##################################################################################
	# weights missing and clustering
	##################################################################################
	non_missing_pcts = np.array([v['sample_row_pct'] + 0.0001 for v in missing_infos.values()])

	# clustering based on sample size
	# non_missing_bins = [0, 0.15, 0.3, 0.6, 0.8, 1.01]
	# n_clusters = len(non_missing_bins) - 1
	# groups = [[] for _ in range(n_clusters)]
	# for idx, non_missing_pct in enumerate(non_missing_pcts):
	# 	for i in range(len(non_missing_bins) - 1):
	# 		if non_missing_bins[i] <= non_missing_pct < non_missing_bins[i + 1]:
	# 			groups[i].append(idx)

	groups = client_groups
	n_clusters = len(groups)

	##################################################################################
	# missing weights
	##################################################################################
	ms_field = 'missing_cell_pct'
	ms_weights = np.array([1 - v[ms_field] + 0.0001 for v in missing_infos.values()])

	###################################################################################
	# Average in-group weights
	###################################################################################
	scale_factor = settings['algo_params']['scale_factor']
	weights_clusters = []
	ms_ratio_clusters = []
	ms_weights_clusters = []
	for i in range(n_clusters):
		if len(groups[i]) == 0:
			continue
		# losses_group = losses[groups[i]]
		# losses_group = losses_group / losses_group.sum()
		ms_weights_w = ms_weights[groups[i]] ** scale_factor
		ms_weights_w = ms_weights_w / ms_weights_w.sum()

		# average in-group weights using losses
		weights_group = weights[groups[i]]
		weight_average = np.average(weights_group, axis=0, weights=ms_weights_w)
		weights_clusters.append(weight_average)

		# average ms_weights
		non_missing_pcts_group = non_missing_pcts[groups[i]]
		ms_ratio_clusters.append(np.average(non_missing_pcts_group, weights=ms_weights_w))
		ms_weight = ms_weights[groups[i]]
		ms_weights_clusters.append(np.average(ms_weight, weights=ms_weights_w))

	weights_avg_clusters = np.array(weights_clusters)
	ms_ratio_avg_clusters = np.array(ms_ratio_clusters) ** scale_factor
	final_weight_average = ms_ratio_avg_clusters

	# take average across clusters
	final_weight_average = np.average(weights_avg_clusters, axis=0, weights=final_weight_average)

	return final_weight_average


def normalization(x, inverse=False, ratio=0.0):
	if inverse:
		x = 1 / np.exp(x)  # 0.0110 0.0111 0.0112
		return x
	x_range = x.max(axis=0) - x.min(axis=0)
	return (x - (x.mean(axis=0) - ratio * x_range)) / ((1 + ratio) * x_range)
