import numpy as np
import random
from typing import List, Dict, Union
from .ms_mech_funcs import mask_sigmoid, mask_quantile


########################################################################################################################
# Sigmoid Based MNAR
########################################################################################################################
def simulate_nan_mnar_sigmoid(
        data: np.ndarray, cols: list, missing_ratio: Union[str, list, dict],
        missing_func: Union[str, list, dict], strict: bool = False, mm_feature_option='all',
        mm_beta_option: str = 'sphere', rng: np.random.Generator = np.random.default_rng(1002031),
) -> np.ndarray:
    """
    sigmoid based MNAR missing values
    :param data: data array
    :param cols:  columns to add missing values
    :param missing_ratio: missing ratio of each column
    :param missing_func: missing mech function of each column
    :param strict: whether to strictly follow the missing ratio or with probability
    :param mm_feature_option: strategies of missing data associated columns
    :param mm_beta_option: how to set beta in logistic function
    :param rng: random generator
    :return: data with missing data - same dimension as data
    """
    mask = np.zeros(data.shape, dtype=bool)

    # add missing for each column
    for idx, col in enumerate(cols):

        # set the seed
        #seed = (seed + 1203941) % (2 ^ 32 - 1)
        # missing is associated with column itself
        #data_corr = data[:, col]

        if isinstance(missing_ratio, dict):
            missing_ratio_ = missing_ratio[col]
        elif isinstance(missing_ratio, list):
            missing_ratio_ = missing_ratio[idx]
        else:
            missing_ratio_ = missing_ratio

        if isinstance(missing_func, dict):
            missing_func_ = missing_func[col]
        elif isinstance(missing_func, list):
            missing_func_ = missing_func[idx]
        else:
            missing_func_ = missing_func

        # set the seed
        #seed = (seed + 1203941) % (2 ^ 32 - 1)

        # missing is associated with column itself
        if mm_feature_option == 'self':
            data_corr = data[:, col]
        elif mm_feature_option == 'all':
            data_corr = data
        elif mm_feature_option.startswith('allk'):
            #np.random.seed(seed)
            k = max(int(float(mm_feature_option.split('allk=')[-1]) * data.shape[1]), 1)
            mi = np.corrcoef(data, rowvar=False)[col]
            mi_idx = np.argsort(mi)[::-1][:k]
            data_corr = data[:, mi_idx]
            if k == 1:
                data_corr = data_corr.reshape(-1, 1)
        else:
            raise NotImplementedError

        #################################################################################
        # pick coefficients and mask missing values
        #################################################################################
        mask = mask_sigmoid(
            mask, col, data_corr, missing_ratio_, missing_func_, strict=strict, mechanism='mnar',
            beta_corr=mm_beta_option, rng = rng
        )

    # assign the missing values
    data_ms = data.copy()
    data_ms[mask] = np.nan

    return data_ms


########################################################################################################################
# Quantile based MNAR
########################################################################################################################
def simulate_nan_mnar_quantile(
        data: np.ndarray, cols: list, missing_ratio: Union[str, list, dict], missing_func: Union[str, list, dict],
        strict: bool = True, rng: np.random.Generator = np.random.default_rng(201030)
) -> np.ndarray:
    """
    Quantile based MNAR missing values
    :param data: data array
    :param cols:  columns to add missing values
    :param missing_ratio: missing ratio of each column
    :param missing_func: missing mech function of each column
    :param strict:  whether to strictly follow the missing ratio or with probability
    :param rng: numpy random generator
    :return: data with missing values - same dimension as data
    """
    # find the columns that are not to be adding missing values
    mask = np.zeros(data.shape, dtype=bool)

    for idx, col in enumerate(cols):

        if isinstance(missing_ratio, dict):
            missing_ratio_ = missing_ratio[col]
        elif isinstance(missing_ratio, list):
            missing_ratio_ = missing_ratio[idx]
        else:
            missing_ratio_ = missing_ratio

        if isinstance(missing_func, dict):
            missing_func_ = missing_func[col]
        elif isinstance(missing_func, list):
            missing_func_ = missing_func[idx]
        else:
            missing_func_ = missing_func

        # set the seed
        # seed = (seed + 10087651) % (2 ** 32 - 1)
        # random.seed(seed)
        # np.random.seed(seed)

        data_corr = data[:, col]

        # find the quantile of the most correlated column
        if missing_func == 'random':
            missing_func = rng.choice(['left', 'right', 'mid', 'tail'])

        # get mask based on quantile
        mask = mask_quantile(mask, col, data_corr, missing_ratio_, missing_func_, strict, rng)

    # assign the missing values
    data_ms = data.copy()
    data_ms[mask] = np.nan

    return data_ms