import numpy as np


def test_aggregate_initial():

	values = [
		[1, 2, 3],
		[9, 4, 3],
		[9, 4, 1],
	]

	missing_ratios = [
		[0.1, 0.9, 0.1],
		[0.1, 0.1, 0.1],
		[0.1, 0.1, 0.9],
	]

	values = np.array(values)
	missing_ratios = 1 - np.array(missing_ratios) + 0.0001
	agg_value = np.zeros(values.shape[1])

	for col in range(values.shape[1]):
		missing_bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
		n_clusters = len(missing_bins) - 1
		groups = [[] for _ in range(n_clusters)]
		for idx, ms_weight in enumerate(missing_ratios[:, col]):
			for i in range(n_clusters):
				if missing_bins[i] <= ms_weight < missing_bins[i + 1]:
					groups[i].append(idx)

		weights_clusters = []
		ms_ratio_clusters = []
		for i in range(n_clusters):
			if len(groups[i]) == 0:
				continue
			# average in-group weights using losses
			weights_group = values[:, col][groups[i]]
			weight_average = np.average(weights_group, axis=0)
			weights_clusters.append(weight_average)
			# average ms_weights
			ms_weights_group = missing_ratios[:, col][groups[i]]
			ms_ratio_clusters.append(ms_weights_group.mean())

		weights_avg_clusters = np.array(weights_clusters)
		ms_ratio_avg_clusters = np.array(ms_ratio_clusters)

		# take average across clusters
		agg_value[col] = np.average(weights_avg_clusters, axis=0, weights=ms_ratio_avg_clusters)

	print(agg_value)
	assert True
