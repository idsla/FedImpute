from numbers import Real
from typing import Any, List, Tuple

import numpy as np
import scipy.stats as stats


MR_RANGE_BANK = {
    'extra-small': (0.1, 0.2),
    'small': (0.2, 0.4),
    'moderate': (0.4, 0.6),
    'large': (0.6, 0.8),
    'extra-large': (0.8, 0.9),
}


def _get_discrete_ratio_values(lower: float, upper: float) -> np.ndarray:
    if lower == upper:
        return np.array([lower])

    step = round((upper - lower) / 0.1) + 1
    return np.linspace(lower, upper, step, endpoint=True)


def _validate_missing_ratio(value: Any) -> float:
    
    if not (isinstance(value, Real) and not isinstance(value, bool)):
        raise ValueError(f'should be a numeric value.')

    value = float(value)
    if value < 0 or value > 1:
        raise ValueError(f'should be in [0, 1].')

    return value


def resolve_ms_mr_clients(ms_mr_clients: Any, num_clients: int) -> List[Tuple[float, float]]:
    """
    Normalize client missing ratio settings to a list of per-client ratio ranges.

    :param ms_mr_clients: scalar, tuple range, or list of scalars/ranges/bucket names
    :param num_clients: number of clients
    :return: list of tuple ranges with length num_clients
    """
    def _resolve_missing_ratio_entry(value: Any) -> Tuple[float, float]:
    
        if isinstance(value, str):
            if value not in MR_RANGE_BANK:
                raise ValueError(f'Unknown missing ratio bucket: {value}.')
            return MR_RANGE_BANK[value]
        
        if isinstance(value, Real) and not isinstance(value, bool):
            ratio = _validate_missing_ratio(value)
            return ratio, ratio

        if isinstance(value, tuple):
            if len(value) != 2:
                raise ValueError('Missing ratio range tuple should be of length 2.')
            lower, upper = value
            lower = _validate_missing_ratio(lower)
            upper = _validate_missing_ratio(upper)
            if lower > upper:
                raise ValueError('Lower bound should be less than or equal to upper bound.')
            return lower, upper

        raise ValueError(
            f'ms_mr_clients should be a numeric value, a ratio range tuple, or a predefined ratio bucket.'
        )
    

    if num_clients <= 0:
        raise ValueError('num_clients should be greater than 0.')

    # if it's already a list, validate each entry and return the list of resolved ranges
    if isinstance(ms_mr_clients, list):
        if len(ms_mr_clients) == 0:
            raise ValueError('ms_mr_clients should not be an empty list.')
        if len(ms_mr_clients) != num_clients:
            raise ValueError('Length of ms_mr_clients should be equal to num_clients.')
        return [
            _resolve_missing_ratio_entry(value)
            for idx, value in enumerate(ms_mr_clients)
        ]
    
    # for scalar or tuple input, apply the same setting to all clients
    else:
        mr_range = _resolve_missing_ratio_entry(ms_mr_clients)
        return [mr_range for _ in range(num_clients)]


def generate_missing_ratios(
    dist: str,
    ms_range: List[Tuple[float, float]],
    num_clients: int,
    num_cols: int,
    seed: int,
    mr_lower: float = 0.1,
    mr_upper: float = 0.9
) -> List[List[float]]:
    """
    Generate missing ratios for each client and each feature
    Options:
    - random: missing ratio is uniformly distributed between each client's range
    - normal: missing ratio is truncated normal distributed with mu=dist_params['mu'] and sigma=dist_params['loc']
    - random-int: missing ratio is uniformly distributed between each client's range with step 0.1
    - normal-int: missing ratio is truncated normal distributed with mu=dist_params['mu'] and sigma=dist_params['loc']

    :param dist: distribution type - support random, random-int, normal, normal-int
    :param ms_range: missing ratio lower and upper bounds for each client
    :param num_clients: number of clients
    :param num_cols:  number of features
    :param seed: seed
    :param mr_lower: final missing ratio lower clipping bound
    :param mr_upper: final missing ratio upper clipping bound
    :return: missing ratios array of shape (num_clients, num_cols)
    """
    
    # validate mr_lower, mr_upper, and ms_range
    mr_lower = _validate_missing_ratio(mr_lower)
    mr_upper = _validate_missing_ratio(mr_upper)
    if mr_lower > mr_upper:
        raise ValueError('mr_lower should be less than or equal to mr_upper.')

    # check num_clients
    if num_clients > 50:
        raise ValueError('In cross silo settings - num_clients should be less than 100')

    # distribution
    rng = np.random.RandomState(seed)
    missing_ratios = np.empty((num_clients, num_cols))
    for client_idx, (lower, upper) in enumerate(ms_range):
        if dist == 'random':
            missing_ratios[client_idx] = rng.uniform(lower, upper, num_cols)
        elif dist == 'random-int':
            mr_list = _get_discrete_ratio_values(lower, upper)
            missing_ratios[client_idx] = rng.choice(mr_list, num_cols)
        elif dist == 'normal':
            if lower == upper:
                missing_ratios[client_idx] = np.ones(num_cols) * lower
            else:
                mu = (lower + upper) / 2
                sigma = (upper - lower) / 3
                trunc_norm_dist = stats.truncnorm(
                    (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
                )
                missing_ratios[client_idx] = trunc_norm_dist.rvs(size=num_cols, random_state=rng)
        elif dist == 'normal-int':
            mr_list = _get_discrete_ratio_values(lower, upper)
            if len(mr_list) == 1:
                missing_ratios[client_idx] = np.ones(num_cols) * mr_list[0]
            else:
                mu = np.mean(mr_list)
                sigma = (upper - lower) / 3
                probs = stats.truncnorm.pdf(
                    mr_list, (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma
                )
                missing_ratios[client_idx] = rng.choice(mr_list, num_cols, p=probs / probs.sum())
        else:
            raise ValueError('Strategy not found')

    missing_ratios = np.clip(missing_ratios, mr_lower, mr_upper)

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
    rng = np.random.RandomState(seed)
    # homogenous missing mechanism distribution
    if mm_dist == "identity":
        missing_mechanism_dist_cols = rng.choice(mm_list, (num_cols,))
        missing_mechanism_dist = [list(missing_mechanism_dist_cols.copy()) for _ in range(num_clients)]
    # random missing mechanism distribution
    elif mm_dist == 'random':
        if len(mm_list) == 1:
            raise ValueError('mm funcs have multiple functions in random case, please use homo in this case.')
        missing_mechanism_dist = rng.choice(mm_list, (num_clients, num_cols)).tolist()
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
                rng.shuffle(col_funcs)
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
