import pandas as pd
import numpy as np
import random


########################################################################################################################
# Base functions to add missing value for one column
########################################################################################################################
def add_missing_single_col(
		series: pd.Series, missing_prob: float = 0.1,
		seed: int = 0
) -> pd.Series:
	"""
	Add missing value with probability to a selection range of the dataframe columns
	:param series: pandas Series
	:param missing_prob: probability of a value to be missing
	:param seed: random seed
	:return: pandas Dataframe with missing value added
	"""
	# set random seed
	random.seed(seed)
	np.random.seed(seed)

	# get random indices
	indices = series.index.tolist()
	indices_sample = random.sample(indices, int(len(indices) * missing_prob), )

	# perform missing
	if len(indices_sample):
		series.loc[pd.Index(indices_sample)] = np.nan

	return series


########################################################################################################################
# Value selection based on conditions
########################################################################################################################
def data_selection_num(series: pd.Series, quantile_range: tuple = (0.8, 1.0)) -> pd.Index:
	"""
	Select indices of subset of numerical series value based on quantile range
	:param series: pd.Series
	:param quantile_range: range defined using quantile
	:return: pd.Index
	"""
	range_lower_qt, range_upper_qt = quantile_range
	if range_lower_qt < 0 or range_upper_qt > 1:
		raise ValueError('Range percentile out of bound 0 - 100')

	if range_lower_qt > range_upper_qt:
		lower_qt = series.quantile(range_lower_qt)
		upper_qt = series.quantile(range_upper_qt)
		series_max = series.max()
		series_min = series.min()
		series = series[((series > lower_qt) & (series < series_max)) | ((series > series_min) & (series < upper_qt))]
	else:
		lower_qt = series.quantile(range_lower_qt)
		upper_qt = series.quantile(range_upper_qt)
		series = series[(series >= lower_qt) & (series <= upper_qt)]

	index = series.index

	return index


########################################################################################################################
# MCAR, and MAR, MNAR
########################################################################################################################
def add_missing_MCAR(
		series: pd.Series, missing_prob: float = 0.2,
		seed: int = 0
) -> pd.Series:
	"""
	Add missing value to a single column completely at random
	:param series: pandas Dataframe
	:param column_name: column name
	:param missing_prob: missing probability
	:param seed: random seed
	:return: new Pandas Dataframe with missing value added
	"""
	new_series = series.copy()
	return add_missing_single_col(new_series, missing_prob, seed)


def add_missing_MNAR(
		series: pd.Series, series_type: str = 'cat', series_values: list = None, missing_probs: list = None,
		seed: int = 0
) -> pd.Series:
	"""
	Add missing value not at random, missing value related to the missing reason
	@param series: current column series
	@param series_type: series data type - num or cat
	@param series_values: series value for each group

	@param missing_probs: missing prob for each group
	@param seed: random seed

	@return new_series: series add missing values
	"""
	if series_values is None:
		series_values = []

	if missing_probs is None:
		missing_probs = []

	new_series = series.copy()

	for value, missing_prob in zip(series_values, missing_probs):
		# print(value, missing_prob)
		if series_type == 'cat':
			index = series[series == value].index
		elif series_type == 'num':
			index = data_selection_num(series, value)
		else:
			raise ValueError('Filter type is not support. Should be one of "cat", "num"')

		# add missing value to selected part of the data
		new_series[index] = add_missing_single_col(new_series[index], missing_prob, seed)

	return new_series


def add_missing_MNAR_s(
		series: pd.Series, sensitive_series: pd.Series, series_type: str = 'cat', sensitive_type: str = 'cat',
		param_values: dict = None, seed: int = 0
) -> pd.Series:
	"""
	Add missing value not at random, missing value related to the missing reason
	@param series: current column series
	@param series_type: series data type - num or cat
	@param sensitive_series: sensitive column series
	@param sensitive_type: sensitive column data type - num or cat
	@param param_values: a dictionary contains how to perform missing
	@param seed: random seed

	@return new_series: series add missing values
	"""
	if param_values is None:
		raise ValueError('None error')

	new_series = series.copy()
	# print(param_values)
	for group, group_params in param_values.items():
		if sensitive_type == 'cat':
			index_group = sensitive_series[sensitive_series == group].index
		elif sensitive_type == 'num':
			index_group = data_selection_num(sensitive_series, group)
		else:
			raise ValueError("corr filter value is not correct")
		# print(group_params[0], group_params[1])
		for value, missing_prob in zip(group_params[0], group_params[1]):
			if series_type == 'cat':
				index = series[index_group][series == value].index
			elif series_type == 'num':
				index = data_selection_num(series[index_group], value)
			else:
				raise ValueError('Filter type is not support. Should be one of "cat", "num"')

			# add missing value to selected part of the data
			new_series[index] = add_missing_single_col(new_series[index], missing_prob, seed)

	return new_series


def add_missing_MAR(
		series: pd.Series, sensitive_series: pd.Series, sensitive_type: str = 'cat',
		sensitive_values: list = None, missing_probs: list = None, seed: int = 0
) -> pd.Series:
	"""
	Add missing value at random, which means missing value is correlated with information in another column
	:param series: series of data
	:param sensitive_series: series of correlated data
	:param sensitive_type: cat or num
	:param sensitive_values: quantile range for numerical data
	:param missing_probs: probability of missing for each value of filter
	:param seed: random seed
	:return: pandas Dataframe
	"""
	if sensitive_values is None:
		sensitive_values = []

	if missing_probs is None:
		missing_probs = []

	new_series = series.copy()
	for filter_value, missing_prob in zip(sensitive_values, missing_probs):
		# select subset of data based on range - cat => categories num => quantile range
		if sensitive_type == 'cat':
			index = sensitive_series[sensitive_series == filter_value].index
		elif filter_value == 'num':
			index = data_selection_num(sensitive_series, filter_value)
		else:
			raise ValueError('Filter type is not support. Should be one of "cat", "num"')

		# perform adding missing value to this subselection of data
		new_series[index] = add_missing_single_col(new_series[index], missing_prob, seed)

	return new_series


def add_missing_MAR_SY(
		series: pd.Series, sensitive_series: pd.Series, y_series:pd.Series, sensitive_y_values:list,
		missing_probs: list = None, seed: int = 0
) -> pd.Series:
	"""
	Add missing value at random, which means missing value is correlated with information in another column
	:param series: series of data
	:param sensitive_series: series of correlated data
	:param y_series: y_series
	:param sensitive_y_values:
	:param missing_probs: probability of missing for each value of filter
	:param seed: random seed
	:return: pandas Dataframe
	"""
	if sensitive_y_values is None:
		raise ValueError('sensitive y values cannot be None')

	if missing_probs is None:
		raise ValueError('missing probs cannot be know')

	new_series = series.copy()
	for (sensitive_value, y_value), missing_prob in zip(sensitive_y_values, missing_probs):
		# select subset of data based on range - cat => categories num => quantile range
		sensitive_index = sensitive_series[sensitive_series == sensitive_value].index
		y_index = y_series[y_series == y_value].index
		index = sensitive_index.intersection(y_index)

		# modify missing prob
		missing_prob = missing_prob

		# perform adding missing value to this subselection of data
		new_series[index] = add_missing_single_col(new_series[index], missing_prob, seed)

	return new_series


########################################################################################################################
def add_missing(
		series: pd.Series, option, corr_series: pd.Series, filter_type: str = 'cat',
		filter_values: list = None, missing_probs: list = None, seed: int = 0
):
	if option == 'MCAR':
		return add_missing_MCAR(series, missing_prob=missing_probs[0], seed=seed)
	elif option == 'MAR':
		return add_missing_MAR(series, corr_series, filter_type, filter_values, missing_probs, seed=seed)
	elif option == "MNAR":
		return add_missing_MNAR(series, filter_type, filter_values, missing_probs, seed=seed)
	else:
		raise ValueError("Missing option need to be one of [MCAR, MAR, MNAR]")
