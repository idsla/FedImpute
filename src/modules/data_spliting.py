import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np


########################################################################################################################
# Split dataset into n folds
########################################################################################################################
def split_train_test(dataset: pd.DataFrame, n_folds=5, seed=0, test_size = 0.1, regression=False):
	"""
	Split dataset into n folds train and test sets
	:param dataset: pandas dataset to split
	:param n_folds: number of folds
	:param seed: random seed
	:return: list of train and test sets
	"""
	# # split into n folds
	# k_fold = StratifiedKFold(n_folds, random_state=seed, shuffle=True)
	# splits = k_fold.split(dataset.drop([dataset.columns[-1]], axis=1), dataset[dataset.columns[-1]])

	# split into train and test sets
	train_test_sets = []
	for i in range(n_folds):
		seed = seed + i*173738
		if regression:
			X_train, X_test, y_train, y_test = \
				train_test_split(
					dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=test_size, random_state=seed)
		else:
			X_train, X_test, y_train, y_test = \
				train_test_split(
					dataset.iloc[:, :-1], dataset.iloc[:, -1], test_size=test_size, random_state=seed, stratify=dataset.iloc[:, -1])
		train_set = pd.concat([X_train, y_train], axis=1).reset_index(drop=True).copy()
		test_set = pd.concat([X_test, y_test], axis=1).reset_index(drop=True).copy()
		train_test_sets.append((train_set, test_set))

	return train_test_sets


########################################################################################################################
# split dataset into n parts using sklearn
########################################################################################################################
def partition_data(dataset: np.ndarray, n_parts=5, seed=0):
	"""
	Split dataset into n folds train and test sets
	:param dataset: pandas dataset to split
	:param n_parts: number of folds
	:param seed: random seed
	:return: list of train and test sets
	"""
	if n_parts == 1:
		return [dataset]
	# split into n folds
	k_fold = StratifiedKFold(n_parts, random_state=seed, shuffle=True)
	#k_fold = StratifiedKFold(n_parts)
	splits = k_fold.split(dataset[:, :-1], dataset[:, -1])

	# split into train and test sets
	datasets, indices = [], []
	for _, test_index in splits:
		indices.append(test_index)
		test_set = dataset[test_index].copy()
		datasets.append(test_set)

	indices_all = np.concatenate(indices)
	indices_all.sort()
	assert np.all(indices_all == np.arange(len(dataset)))

	return datasets
