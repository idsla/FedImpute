import numpy as np
import fedimpute.simulator.missing_simulate.mcar_simulate as mcar_simulate
import fedimpute.simulator.missing_simulate.mar_simulate as mar_simulate
import fedimpute.simulator.missing_simulate.mnar_simulate as mnar_simulate
from typing import List, Union
from .add_missing_utils import (
    generate_missing_cols, generate_missing_ratios, generate_missing_mech_funcs
)
import loguru


# TODO: make this to be a class
def add_missing(
        clients_data: List[np.ndarray],
        cols: List[int],
        rngs: List[np.random.Generator],
        obs_cols: List[int],
        global_missing: bool = False,
        mf_strategy: str = 'all',
        mf_dist: str = 'identity',
        mr_dist: str = 'randu',
        mr_lower: float = 0.3,
        mr_upper: float = 0.7,
        mm_funcs_dist: str = 'identity',
        mm_funcs_bank: str = 'lr',
        mm_mech: str = 'mcar',
        mm_strictness: bool = True,
        mm_obs: bool = True,
        mm_feature_option: str = 'allk=0.2',
        mm_beta_option: str = None,
        seed: int = 20030331,
) -> List[np.array]:
    """
    Simulate Missing data
    :param rngs: random generator for each client
    :param clients_data: List of clients data
    :param cols: columns to be considered
    :param seeds: seed for each client
    :param obs_cols: observed columns index
    :param global_missing: whether simulate missing data globally or locally
    :param mf_strategy: missing features strategy
    :param mf_dist: missing features distribution across clients
    :param mr_dist: missing ratio distribution
    :param mr_lower: missing ratio lower bound
    :param mr_upper: missing ratio upper bound
    :param mm_funcs_dist: missing mechanism functions distribution across clients and features
    :param mm_funcs_bank: missing mechanism functions banks
    :param mm_mech: missing mechanism
    :param mm_strictness: missing adding probailistic or deterministic
    :param mm_obs:  missing adding based on observed data
    :param mm_feature_option: missing mechanism associated with which features
    :param mm_beta_option: missing mechanism beta coefficient option
    :param seed: randomness
    :return: list of datasets with missing values
    """

    # remove observed columns from ms_cols
    if not mm_mech.startswith('mar'):
        mm_obs = False

    if mm_obs:
        if len(obs_cols) > 0:
            cols = np.array(cols)
            cols = cols[~np.isin(cols, obs_cols)].tolist()

    # add missing to global data then split
    if global_missing:
        global_data = np.concatenate(clients_data, axis=0)
        split_indices = np.cumsum([item.shape[0] for item in clients_data])[:-1]

        global_data_ms = _add_missing_central(
            global_data, cols, mf_strategy=mf_strategy, mr_dist=mr_dist, mr_lower=mr_lower, mr_upper=mr_upper,
            mm_funcs_bank=mm_funcs_bank, mm_mech=mm_mech, mm_strictness=mm_strictness, mm_obs=mm_obs,
            mm_feature_option=mm_feature_option, mm_beta_option=mm_beta_option, seed=seed
        )

        clients_data_ms = np.array_split(global_data_ms, split_indices)

    # add missing to separately
    else:
        clients_data_ms = _add_missing_dist(
            clients_data, cols, rngs, mf_strategy=mf_strategy, mr_dist=mr_dist, mr_lower=mr_lower, mr_upper=mr_upper,
            mm_funcs_dist=mm_funcs_dist, mm_funcs_bank=mm_funcs_bank, mm_mech=mm_mech, mm_strictness=mm_strictness,
            mm_obs=mm_obs, mm_feature_option=mm_feature_option, mm_beta_option=mm_beta_option, seed=seed
        )

    return clients_data_ms


########################################################################################################################
def _add_missing_central(
        data: np.ndarray, cols: List[int],
        mf_strategy: str = 'all',
        mf_dist: str = 'identity',
        mr_dist: str = 'randu',
        mr_lower: float = 0.3,
        mr_upper: float = 0.7,
        mm_funcs_bank: str = 'lr',
        mm_mech: str = 'mcar',
        mm_strictness: bool = True,
        mm_obs: bool = True,
        mm_feature_option: str = 'allk=0.2',
        mm_beta_option: Union[str, None] = None,
        seed: int = 102031
) -> np.ndarray:
    """
    Add missing data for global dataset
    :param data: data array
    :param cols: columns to add missing values
    :param mm_funcs_bank: missing mechanism functions banks
    :param mf_strategy: missing features strategy
    :param mf_dist: missing features distribution across clients
    :param mr_dist: missing ratio distribution
    :param mr_lower: missing ratio lower bound
    :param mr_upper: missing ratio upper bound
    :param mm_mech: missing mechanism
    :param mm_strictness: missing adding probailistic or deterministic
    :param mm_obs:  missing adding based on observed data
    :param mm_feature_option: missing mechanism associated with which features
    :param mm_beta_option: missing mechanism beta coefficient option
    :param seed: randonness
    :return: dataset with missing values
    """

    # missing features - e.g. 'all'
    # missing_cols_strategy, missing_cols_params = parse_strategy_params(missing_cols_strategy)
    missing_cols = generate_missing_cols(mf_strategy, 1, cols, seed=seed)
    missing_cols = missing_cols[0]
    num_cols = len(missing_cols)

    # missing ratios - 'uniform@mrl=0.3-mrr=0.7'
    # mr_dist, mr_params = parse_strategy_params(mr_strategy)
    mr_range = (mr_lower, mr_upper)
    missing_ratios = generate_missing_ratios(
        mr_dist, ms_range=mr_range, num_clients=1, num_cols=num_cols, seed=seed
    )
    missing_ratios = missing_ratios[0]

    # missing mechanism funcs - e.g. 'mcar'
    # TODO: currently only support all cols with same misisng mechs
    # mm_mech, mm_mech_params = parse_strategy_params(missing_mech)
    # missing_mechs = generate_missing_mech(mm_mech, num_clients=1, num_cols=num_cols, seed=seed)
    # missing mechanism functions - e.g. 'lr'
    mm_funcs = generate_missing_mech_funcs(
        'identity', mm_funcs_bank, num_clients=1, num_cols=num_cols, seed=seed
    )
    mm_funcs = mm_funcs[0]

    loguru.logger.debug(f"{missing_cols} {missing_ratios} {mm_funcs}")

    # Simulate missing data
    X_train, y_train = data[:, :-1], data[:, -1]
    rng = np.random.default_rng(seed)
    X_train_ms = simulate_nan(
        X_train, y_train, mm_mech=mm_mech, missing_features=missing_cols, missing_ratios=missing_ratios,
        mechanism_funcs=mm_funcs, mm_obs=mm_obs, mm_strictness=mm_strictness,
        mm_feature_option=mm_feature_option, mm_beta_option=mm_beta_option, rng=rng
    )

    return X_train_ms


########################################################################################################################
def _add_missing_dist(
        clients_data: List[np.ndarray], cols: List[int], rngs: List[np.random.Generator],
        mf_strategy: str = 'all',
        mf_dist: str = 'identity',
        mr_dist: str = 'randu',
        mr_lower: float = 0.3,
        mr_upper: float = 0.7,
        mm_funcs_dist: str = 'random',
        mm_funcs_bank: str = 'lr',
        mm_mech: str = 'mcar',
        mm_strictness: bool = True,
        mm_obs: bool = True,
        mm_feature_option: str = 'allk=0.2',
        mm_beta_option: str = None,
        seed: int = 20030331
) -> List[np.ndarray]:
    """
        Add missing data for each client's dataset
        :param clients_data: List of data array for clients
        :param cols: columns to add missing values
        :param rngs: list random generator for missing data simulation for each client
        :param mm_funcs_dist: missing mechanism functions distribution across clients and features
        :param mm_funcs_bank: missing mechanism functions banks
        :param mf_strategy: missing features strategy
        :param mf_dist: missing features distribution across clients
        :param mr_dist: missing ratio distribution
        :param mr_lower: missing ratio lower bound
        :param mr_upper: missing ratio upper bound
        :param mm_mech: missing mechanism
        :param mm_strictness: missing adding probailistic or deterministic
        :param mm_obs:  missing adding based on observed data
        :param mm_feature_option: missing mechanism associated with which features
        :param mm_beta_option: missing mechanism beta coefficient option
        :param seed: randonness
        :return: list of datasets with missing values
        """

    num_clients = len(rngs)

    # missing features - e.g. 'all'
    clients_missing_cols = generate_missing_cols(mf_strategy, num_clients, cols, seed=seed)
    num_cols = len(clients_missing_cols[0])

    # missing ratios - 'uniform@mrl=0.3-mrr=0.7'
    # mr_dist, mr_params = parse_strategy_params(mr_strategy)
    mr_range = (mr_lower, mr_upper)
    clients_missing_ratios = generate_missing_ratios(
        mr_dist, ms_range=mr_range, num_clients=num_clients, num_cols=num_cols, seed=seed
    )

    # missing mechanism funcs - 'identity', 'random'
    clients_mm_funcs = generate_missing_mech_funcs(
        mm_funcs_dist, mm_funcs_bank, num_clients=num_clients, num_cols=num_cols, seed=seed
    )

    # Simulate missing data
    assert len(clients_missing_cols) == len(clients_missing_ratios) == len(
        clients_mm_funcs), "error when generate missing data"

    clients_data_ms = []
    for i in range(num_clients):
        missing_cols = clients_missing_cols[i]
        missing_ratios = clients_missing_ratios[i]
        mm_funcs = clients_mm_funcs[i]

        loguru.logger.debug(f"{i} {missing_cols} {missing_ratios} {mm_funcs}")

        X_train, y_train = clients_data[i][:, :-1], clients_data[i][:, -1]
        X_train_ms = simulate_nan(
            X_train, y_train, mm_mech=mm_mech, missing_features=missing_cols, missing_ratios=missing_ratios,
            mechanism_funcs=mm_funcs, mm_obs=mm_obs, mm_strictness=mm_strictness,
            mm_feature_option=mm_feature_option, mm_beta_option=mm_beta_option, rng=rngs[i]
        )

        clients_data_ms.append(X_train_ms)

    return clients_data_ms


########################################################################################################################
# Simulate missing for one client
########################################################################################################################
def simulate_nan(
        X_train: np.ndarray, y_train: np.ndarray, mm_mech: str,
        missing_features: List[int], missing_ratios: List[float], mechanism_funcs: List[str],
        mm_strictness: bool, mm_obs: bool, mm_feature_option: str, mm_beta_option: str,
        rng: np.random.Generator = np.random.default_rng(100203)
) -> np.ndarray:
    """
    Simulate missing values for one client
    :param X_train: X_train data
    :param y_train: y_train data
    :param mm_mech: missing mechanism
    :param missing_features: missing features
    :param missing_ratios: missing ratios for each feature
    :param mechanism_funcs: missing mechanism functions for each feature
    :param mm_strictness: missing strictness
    :param mm_obs: missing based on observed data
    :param mm_feature_option: missing mechanism associated with which features
    :param mm_beta_option: missing mechanism beta coefficient option
    :param rng: randomness generator
    :return: data with missing values
    """

    if mm_mech == 'mcar':
        X_train_ms = mcar_simulate.simulate_nan_mcar(
            X_train, missing_features, missing_ratios, rng=rng
        )
    elif mm_mech == 'mar_quantile':
        X_train_ms = mar_simulate.simulate_nan_mar_quantile(
            X_train, missing_features, missing_ratio=missing_ratios, missing_func=mechanism_funcs,
            obs=mm_obs, strict=mm_strictness, rng=rng
        )
    elif mm_mech == 'mar_logit':
        X_train_ms = mar_simulate.simulate_nan_mar_sigmoid(
            X_train, missing_features, missing_ratio=missing_ratios, missing_func=mechanism_funcs,
            obs=mm_obs, strict=mm_strictness, mm_feature_option=mm_feature_option, mm_beta_option=mm_beta_option,
            rng=rng
        )
    elif mm_mech == 'mnar_quantile':
        X_train_ms = mnar_simulate.simulate_nan_mnar_quantile(
            X_train, missing_features, missing_ratio=missing_ratios, missing_func=mechanism_funcs, strict=mm_strictness,
            rng=rng
        )
    elif mm_mech == 'mnar_logit':
        X_train_ms = mnar_simulate.simulate_nan_mnar_sigmoid(
            X_train, missing_features, missing_ratio=missing_ratios, missing_func=mechanism_funcs,
            strict=mm_strictness, mm_feature_option=mm_feature_option, mm_beta_option=mm_beta_option, rng=rng
        )
    elif mm_mech == 'mnar_sm_logit':
        X_train_ms = mnar_simulate.simulate_nan_mnar_sigmoid(
            X_train, missing_features, missing_ratio=missing_ratios, missing_func=mechanism_funcs,
            strict=mm_strictness, mm_feature_option='self', mm_beta_option=mm_beta_option, rng=rng
        )
    else:
        raise NotImplementedError

    return X_train_ms
