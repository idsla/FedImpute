import pandas as pd
from typing import Union, List, Tuple
import random
import numpy as np
from dataclasses import dataclass


@dataclass
class MissingParamsVar:
    """
    Dataclass to store missing parameters for each variable
    """
    var_idx: int
    var_type: str
    var_filter_range: Union[List, Tuple]


########################################################################################################################
# add missing MCAR
########################################################################################################################
def add_missing_single_column_mcar(
        series: pd.Series, missing_prob: float = 0.2, seed: int = 0
) -> pd.Series:
    """
    Add missing value to a single column completely at random
    """
    new_series = series.copy()
    return add_missing_single_col(new_series, missing_prob, seed)


########################################################################################################################
# add missing MAR
########################################################################################################################
def add_missing_single_column_mar(
        series: pd.Series, associated_vars_data, associated_vars: List[MissingParamsVar], missing_prob: float,
        seed: int = 0
) -> pd.Series:
    """
    Add missing value at random, missing value related to the missing reason
    """
    new_series = series.copy()
    missing_prob_each_var = missing_prob / len(associated_vars)  # assign missing prob to each associated vars

    # add missing value to each associated vars
    for idx, var_params in enumerate(associated_vars):
        series_type = var_params.var_type
        series_range = var_params.var_filter_range
        if associated_vars_data.ndim == 1:
            associated_series = associated_vars_data
        else:
            associated_series = associated_vars_data.iloc[:, idx]

        index = get_index_by_range(associated_series, series_type, series_range)

        # add missing value to selected part of the data
        new_series[index] = add_missing_single_col(new_series[index], missing_prob_each_var, seed)

    return new_series


########################################################################################################################
# add missing MNAR
########################################################################################################################
def add_missing_single_column_mnar(
        series: pd.Series, series_type: str = 'cat', range: Union[List, Tuple] = None,
        missing_prob: float = None, seed: int = 0
) -> pd.Series:
    """
    Add missing value not at random, missing value related to the missing reason
    """
    new_series = series.copy()

    # get index of selected range
    index = get_index_by_range(new_series, series_type, range)

    # add missing value to selected part of the data
    new_series[index] = add_missing_single_col(new_series[index], missing_prob, seed)

    return new_series


##########################################################################################################
# Utils
##########################################################################################################
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


def get_index_by_range(series, series_type, value_range=None):
    if series_type == 'cat':
        if value_range is None:
            value_range = series.value_counts().index.tolist()[0]
        index = series[series.isin(value_range)].index
    elif series_type == 'num':
        if value_range is None:
            value_range = (0.5, 1)
        index = data_selection_num(series, value_range)
    else:
        raise ValueError('Filter type is not support. Should be one of "cat", "num"')

    return index
