from typing import Tuple, List

import numpy as np
import scipy.stats as stats


def generate_missing_ratios(
        dist: str, ms_range: Tuple[float, float], num_clients: int, num_cols: int, seed: int
) -> List[List[float]]:
    """
    Generate missing ratios for each client and each feature
    Options:
    - fixed: missing ratio is fixed for all clients and features
    - uniform: missing ratio is uniformly distributed between ms_range[0] and ms_range[1]
    - gaussian: missing ratio is truncated normal distributed with mu=dist_params['mu'] and sigma=dist_params['loc']
    - uniform_int: missing ratio is uniformly distributed between ms_range[0] and ms_range[1] with step 0.1
    - gaussian_int: missing ratio is truncated normal distributed with mu=dist_params['mu'] and sigma=dist_params['loc']

    :param dist: distribution type - support fixed, uniform, gaussian, uniform_int
    :param ms_range: missing ratios upper and lower bounds
    :param num_clients: number of clients
    :param num_cols:  number of features
    :param seed: seed
    :return: missing ratios array of shape (num_clients, num_cols)
    """

    # check missing ratios
    if ms_range[0] > ms_range[1]:
        raise ValueError('ms_range[0] should be less than ms_range[1]')
    elif ms_range[0] < 0 or ms_range[0] > 1:
        raise ValueError('ms_range[0] should be in [0, 1]')
    elif ms_range[1] < 0 or ms_range[1] > 1:
        raise ValueError('ms_range[1] should be in [0, 1]')

    # check num_clients
    if num_clients > 50:
        raise ValueError('In cross silo settings - num_clients should be less than 100')

    # distribution
    np.random.seed(seed)
    if dist == 'fixed':
        missing_ratios = np.ones((num_clients, num_cols)) * ms_range[0]
    elif dist == 'randu':
        missing_ratios = np.random.uniform(ms_range[0], ms_range[1], (num_clients, num_cols))
    elif dist == 'randu-int':
        start = float(ms_range[0])
        stop = float(ms_range[1])
        step = round((stop - start) / 0.1) + 1
        mr_list = np.linspace(start, stop, step, endpoint=True)
        missing_ratios = np.random.choice(mr_list, (num_clients, num_cols))
    elif dist == 'randn':
        lower, upper = ms_range[0], ms_range[1]
        mu, sigma = (lower + upper) / 2, (upper - lower) / 3  # sigma set to range/3 - 1.5 std cover range
        # truncated norm distribution
        trunc_norm_dist = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        missing_ratios = trunc_norm_dist.rvs(size=(num_clients, num_cols))
    elif dist == 'randn-int':
        start = float(ms_range[0])
        stop = float(ms_range[1])
        step = round((stop - start) / 0.1) + 1
        mr_list = np.linspace(start, stop, step, endpoint=True)
        # truncated norm distribution
        lower, upper = ms_range[0], ms_range[1]
        mu, sigma = np.mean(mr_list), (upper - lower) / 3  # mean and sigma - 1.5 std = 1/2(upper - lower)
        probs = stats.truncnorm.pdf(mr_list, (lower - mu) / sigma, (upper - mu) / sigma, loc=0.5, scale=sigma)
        missing_ratios = np.random.choice(mr_list, (num_clients, num_cols), p=probs / probs.sum())
    else:
        raise ValueError('Strategy not found')

    return missing_ratios.tolist()


def generate_missing_mech(mm_mech: str, num_clients: int, num_cols: int, seed: int) -> List[List[str]]:
    """
    Generate missing mechanism for each client and each feature
    TODO: supports more scenarios e.g. mixed missing mechanism one for each client
    :param mm_mech: missing mechanism type str - one of mcar, marq, marqst, marsig, marsigst, mnarq, mnarqst, mnarsig, marsigst
    :param num_clients: number of clients
    :param num_cols: number of features
    :param seed: random seed
    :return: List[List[str]] - missing mechanism for each client and each feature - (num_clients, num_cols)
    """
    np.random.seed(seed)
    MECH_MAPPING = {
        'mcar': 'mcar',
        'marq': 'mar_quantile',
        'marqst': 'mar_quantile_strict',
        'marlogit': 'mar_logit',
        'marlogitst': 'mar_logit_strict',
        'mnarq': 'mnar_quantile',
        'mnarqst': 'mnar_quantile_strict',
        'mnarlogit': 'mnar_logit',
        'mnarlogitst': 'mnar_logit_strict',
        'mnarsmlogit': 'mnar_sm_logit',
        'mnarsmlogitst': 'mnar_sm_logit_strict'
    }

    if mm_mech not in MECH_MAPPING:
        raise ValueError(f'{mm_mech} not supported.')
    else:
        mm_mech = MECH_MAPPING[mm_mech]
        missing_mech_types = [[mm_mech for _ in range(num_cols)] for _ in range(num_clients)]

    return missing_mech_types


def generate_missing_funcs_list(mm_funcs: str) -> List[str]:
    """
    Based on mm_funcs generate missing mechanism functions
    :param mm_funcs:
    :return:
    """

    if mm_funcs is None:
        mm_list = [None]
    elif mm_funcs == 'lr':
        mm_list = ['left', 'right']
    elif mm_funcs == 'mt':
        mm_list = ['mid', 'tail']
    elif mm_funcs == 'all':
        mm_list = ['left', 'right', 'mid', 'tail']
    elif mm_funcs == 'l':
        mm_list = ['left']
    elif mm_funcs == 'r':
        mm_list = ['right']
    elif mm_funcs == 'm':
        mm_list = ['mid']
    elif mm_funcs == 't':
        mm_list = ['tail']
    else:
        raise ValueError(f'mm not found, params: {mm_funcs}')

    return mm_list


def generate_missing_mech_funcs(
        mm_dist: str, mm_funcs, num_clients: int, num_cols: int, seed: int
) -> List[List[str]]:
    """
    Generate missing mechanism distribution for each client and each feature
    :param mm_dist: missing mechanism distribution type - homo, random, random2
    :param mm_funcs: missing funcs list (column missing mechanism funcs banks)
    :param num_clients: number of clients
    :param num_cols: number of features
    :param seed: random seed
    :return: missing mechanism funcs array - (num_client, num_cols)
    """

    mm_list = generate_missing_funcs_list(mm_funcs)
    np.random.seed(seed)
    # homogenous missing mechanism distribution
    if mm_dist == "identity":
        missing_mechanism_dist_cols = np.random.choice(mm_list, (num_cols,))
        missing_mechanism_dist = [list(missing_mechanism_dist_cols.copy()) for _ in range(num_clients)]
    # random missing mechanism distribution
    elif mm_dist == 'random':
        if len(mm_list) == 1:
            raise ValueError('mm funcs have multiple functions in random case, please use homo in this case.')
        missing_mechanism_dist = np.random.choice(mm_list, (num_clients, num_cols)).tolist()
    # random missing mechanism by shuffling
    elif mm_dist == 'random2':
        if len(mm_list) == 1:
            raise ValueError('mm funcs have multiple functions in random case, please use homo in this case.')

        if len(mm_list) == 2:
            N1 = num_clients // 2
            N2 = num_clients - N1
            col_funcs = np.array([mm_list[0]] * N1 + [mm_list[1]] * N2)
            ret = np.empty((num_clients, num_cols), dtype='U5')
            for col in range(num_cols):
                np.random.shuffle(col_funcs)
                ret[:, col] = col_funcs.copy()

            missing_mechanism_dist = ret.tolist()
        else:
            raise NotImplementedError
    else:
        raise ValueError

    return missing_mechanism_dist


def generate_missing_cols(
        missing_col_strategy: str, num_clients: int, cols: List[int], seed: int = 20103
) -> List[List[int]]:
    """
    Generate features indices to add missing values
    :param missing_col_strategy: missing column strategies
    :param num_clients: number of clients
    :param cols: feature indices list
    :param seed: random seed
    :return: list of missing feature indices for each client - (num_client, num_cols)
    """

    if missing_col_strategy == 'all':
        cols = np.expand_dims(np.array(cols), 0)
        return np.repeat(cols, num_clients, axis = 0).tolist()
    else:
        raise NotImplementedError
