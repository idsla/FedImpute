from typing import Union

import numpy as np
import random

from .ms_mech_funcs import mask_sigmoid, mask_quantile


########################################################################################################################
# Quantile based MAR simulation
def simulate_nan_mar_quantile(
        data, cols, missing_ratio, missing_func='left', obs=False, strict=True, rng=np.random.default_rng(201030)
):
    """
    Simulate missing values for MAR mechanism using quantile based method
    :param data: data
    :param cols: columns to add missing values
    :param missing_ratio: missing ratio for each column
    :param missing_func: misisng mech function for each column
    :param obs: whether to add missing values based on only set of observed columns or all columns
    :param strict: strictly add missing values or with probability
    :param rng: random generator
    :return: data with missingness added - same dimension of data
    """
    # find the columns that are not to be adding missing values
    mask = np.zeros(data.shape, dtype=bool)

    obs_cols = []
    for index in range(data.shape[1]):
        if index not in cols:
            obs_cols.append(index)
    obs_cols = np.array(obs_cols)
    if obs and len(obs_cols) == 0:
        raise ValueError("No columns to observe, try to set obs to be False")

    for idx, col in enumerate(cols):

        ##########################################################
        # random seed
        # seed = (seed + 10087651) % (2 ** 32 - 1)
        # np.random.seed(seed)
        # random.seed(seed)

        ##########################################################
        # fetch missing ratio and missing func
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

        if missing_func_ == 'random':
            missing_func_ = random.choice(['left', 'right', 'mid', 'tail'])

        ##########################################################
        # select one most correlated column
        if obs:
            X_rest = data[:, obs_cols]
        else:
            X_rest = data[:, np.arange(data.shape[1]) != col]

        X = np.concatenate([data[:, col].reshape(-1, 1), X_rest], axis=1)
        mi = np.abs(np.corrcoef(X, rowvar=False)[0])
        mi_idx = np.argsort(mi)[::-1][1]
        data_corr = X[:, mi_idx]

        ##########################################################
        # add missing value to mask
        mask = mask_quantile(
            mask, col, data_corr, missing_ratio=missing_ratio_, missing_func=missing_func_, strict=strict, rng=rng
        )

    # assign the missing values
    data_ms = data.copy()
    data_ms[mask] = np.nan

    return data_ms


# def simulate_nan_mary_quantile(
#         data, cols, missing_ratio, missing_func='left', strict=True, seed=201030
# ):
#     mask = mask_mar_quantile(
#         data, cols, obs_cols=None, ms_ratio=missing_ratio, missing_func=missing_func, obs=False,
#         seed=seed, strict=strict, y=True
#     )
#
#     # assign the missing values
#     data_ms = data.copy()
#     data_ms[mask] = np.nan
#
#     return data_ms


########################################################################################################################
# Sigmoid based MAR simulation
def simulate_nan_mar_sigmoid(
        data: np.ndarray, cols: list, missing_ratio: Union[str, list, dict], missing_func: Union[str, list, dict],
        strict: bool = False, obs: bool = False, mm_feature_option='all', mm_beta_option: str = 'random_uniform',
        rng: np.random.Generator = np.random.default_rng(1002031)
) -> np.ndarray:
    """
    Simulate missing values for MAR mechanism using sigmoid function
    :param data: data
    :param cols: columns to add missing values
    :param missing_ratio: missing ratio of each column
    :param missing_func: missing function of each column
    :param strict: whether to strictly add missing or missing with probability
    :param obs: whether to add missing values based on only set of observed columns or all columns
    :param mm_feature_option: how to select missing data with associated columns
    :param mm_beta_option: how to set coefficients beta of associated columns
    :param rng: numpy random generator
    :return: data with missingness added - same dimension of data
    """
    mask = np.zeros(data.shape, dtype=bool)

    # add missing for each column
    for idx, col in enumerate(cols):

        ##################################################################################
        # set the seed and missing ratio and funcs
        #seed = (seed + 1203941) % (2 ^ 32 - 1)
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

        ##################################################################################
        # na associated with only observed columns
        if obs is True:
            keep_mask = np.ones(data.shape[1], dtype=bool)
            keep_mask[list(cols)] = False
            X_rest = data[:, keep_mask]
        # na associated with all other columns
        else:
            X_rest = data[:, np.arange(data.shape[1]) != col]

        #################################################################################
        # get k most correlated columns or all columns
        if mm_feature_option == 'all':
            data_corr = X_rest
        elif mm_feature_option.startswith('allk'):
            #np.random.seed(seed)
            X = np.concatenate([data[:, col].reshape(-1, 1), X_rest], axis=1)
            k = max(int(float(mm_feature_option.split('allk=')[-1]) * X_rest.shape[1]), 1)
            k = min(k, X_rest.shape[1])
            mi = np.abs(np.corrcoef(X, rowvar=False)[0])
            mi_idx = np.argsort(mi)[::-1][1:k + 1]
            data_corr = X[:, mi_idx]
            if k == 1:
                data_corr = data_corr.reshape(-1, 1)
        else:
            raise NotImplementedError

        #################################################################################
        # pick coefficients and mask missing values
        #################################################################################
        mask = mask_sigmoid(
            mask, col, data_corr, missing_ratio_, missing_func_, strict=strict, mechanism='mar',
            beta_corr=mm_beta_option, rng=rng
        )

    # assign the missing values
    data_ms = data.copy()
    data_ms[mask] = np.nan

    return data_ms

# def simulate_nan_mary_sigmoid(data, cols, missing_ratio, missing_func, strict=False, seed=1002031):
#     mask = np.zeros(data.shape, dtype=bool)
#
#     # add missing for each columns
#     for col in cols:
#
#         # set the seed
#         seed = (seed + 1203941) % (2 ^ 32 - 1)
#         # missing is associated with last column is the target
#         indices_obs = [data.shape[1] - 1]
#         data_corr = data[:, indices_obs]
#
#         if data_corr.ndim == 1:
#             data_corr = data_corr.reshape(-1, 1)
#
#         #################################################################################
#         # pick coefficients and mask missing values
#         #################################################################################
#         mask = mask_mar_sigmoid(mask, col, data_corr, missing_ratio, missing_func, strict, seed)
#
#     # assign the missing values
#     data_ms = data.copy()
#     data_ms[mask] = np.nan
#
#     return data_ms

########################################################################################################################
# mask sigmoid
########################################################################################################################
# def mask_mar_sigmoid(mask, col, data_corr, missing_ratio, missing_func, strict, seed):
#     np.random.seed(seed)
#     random.seed(seed)
#     #################################################################################
#     # pick coefficients
#     #################################################################################
#     # Pick coefficients so that W^Tx has unit variance (avoids shrinking)
#     if isinstance(missing_ratio, dict):
#         missing_ratio = missing_ratio[col]
#     else:
#         missing_ratio = missing_ratio
#
#     # copy data and do min-max normalization
#     data_copy = data_corr.copy()
#     data_copy = (data_copy - data_copy.min(0, keepdims=True)) / (
#             data_copy.max(0, keepdims=True) - data_copy.min(0, keepdims=True))
#     data_copy = (data_copy - data_copy.mean(0, keepdims=True)) / data_copy.std(0, keepdims=True)
#
#     coeffs = np.random.rand(data_copy.shape[1], 1)
#     # print(coeffs)
#     Wx = data_copy @ coeffs
#     # print(Wx)
#     wss = (Wx) / np.std(Wx, 0, keepdims=True)
#
#     def f(x: np.ndarray) -> np.ndarray:
#         if missing_func == 'left':
#             return expit(-wss + x).mean().item() - missing_ratio
#         elif missing_func == 'right':
#             return expit(wss + x).mean().item() - missing_ratio
#         elif missing_func == 'mid':
#             return expit(np.absolute(wss) - 0.75 + x).mean().item() - missing_ratio
#         elif missing_func == 'tail':
#             return expit(-np.absolute(wss) + 0.75 + x).mean().item() - missing_ratio
#         else:
#             raise NotImplementedError
#
#     intercept = optimize.bisect(f, -50, 50)
#
#     if missing_func == 'left':
#         ps = expit(-wss + intercept)
#     elif missing_func == 'right':
#         ps = expit(wss + intercept)
#     elif missing_func == 'mid':
#         ps = expit(-np.absolute(wss) + 0.75 + intercept)
#     elif missing_func == 'tail':
#         ps = expit(np.absolute(wss) - 0.75 + intercept)
#     else:
#         raise NotImplementedError
#
#     # strict false means using random simulation
#     if strict is False:
#         ber = np.random.binomial(n=1, size=mask.shape[0], p=ps.flatten())
#         mask[:, col] = ber
#     # strict mode based on rank on calculated probability, strictly made missing
#     else:
#         ps = ps.flatten()
#         # print(ps)
#         end_value = np.sort(ps)[::-1][int(missing_ratio * data_copy.shape[0])]
#         indices = np.where((ps - end_value) > 1e-3)[0]
#         # print(len(indices), int(missing_ratio*data_copy.shape[0]), len(np.where(np.absolute(ps - end_value) <=
#         # 1e-3)[0]))
#         if len(indices) < int(missing_ratio * data_copy.shape[0]):
#             end_indices = np.where(np.absolute(ps - end_value) <= 1e-3)[0]
#             end_indices = np.random.choice(
#                 end_indices, int(missing_ratio * data_copy.shape[0]) - len(indices), replace=False
#             )
#             indices = np.concatenate((indices, end_indices))
#         elif len(indices) > int(missing_ratio * data_copy.shape[0]):
#             indices = np.random.choice(indices, int(missing_ratio * data_copy.shape[0]), replace=False)
#
#         mask[indices, col] = True
#
#     return mask
