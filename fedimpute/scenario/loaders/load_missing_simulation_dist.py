from src.modules.missing_simulate.add_missing import simulate_nan
from typing import List, Tuple, Dict, Union, Any
import numpy as np
from emf.params_utils import parse_strategy_params
import scipy.stats as stats

from src.modules.missing_simulate.add_missing_utils import (
    generate_missing_ratios, generate_missing_mech, generate_missing_mech_funcs
)


def add_missing(
        clients_data_list: List[np.ndarray], mm_strategy: str, cols: list, seeds: list
) -> List[np.ndarray]:
    ret = missing_scenario(len(clients_data_list), cols, mm_strategy, seed=seeds[0])

    clients_data_ms = []
    for i in range(len(clients_data_list)):
        data = clients_data_list[i]
        X_train, y_train = data[:, :-1], data[:, -1]
        missing_ratio = ret[i]['missing_ratio']
        missing_mech_func = ret[i]['missing_mech_func']
        missing_mech_type = ret[i]['missing_mech_type']
        missing_feature = ret[i]['missing_features']
        seed = seeds[i]
        X_train_ms = simulate_nan(
            X_train, y_train, missing_features=missing_feature, mech_type=missing_mech_type, missing_ratios=missing_ratio,
            mechanism_funcs=missing_mech_func, seed=seed
        )
        clients_data_ms.append(np.concatenate([X_train_ms, y_train.reshape(-1, 1)], axis=1).copy())

    return clients_data_ms



def missing_scenario(n_clients: int, cols: list, mm_strategy: str, seed: int = 201030) -> List[dict]:
    mm_strategy, mm_params = parse_strategy_params(mm_strategy)

    ####################################################################################################################
    # Simulate Missing Ratios for each client and each features
    ####################################################################################################################
    # mr_strategy: fixed@mrl=0.1-mrr=0.9
    mr_dist, mr_params = parse_strategy_params(mm_params['mr_strategy'])
    mr_range = (float(mr_params['mrl']), float(mr_params['mrr']))
    missing_ratios = generate_missing_ratios(mr_dist, mr_range, n_clients, len(cols), seed)

    ####################################################################################################################
    # Simulate Missing Mechanism
    ####################################################################################################################
    # {'mm_dist': '', 'mm_mech': '', 'mm_funcs': ''}
    if 'mm_dist' not in mm_params or 'mm_mech' not in mm_params or 'mm_funcs' not in mm_params:
        raise ValueError('mm_params should contain "mm_dist", "mm_mech", "mm_funcs" keys.')

    mm_dist, mm_mech, mm_funcs = mm_params['mm_dist'], mm_params['mm_mech'], mm_params['mm_funcs']

    # mechanisms
    missing_mech_types = generate_missing_mech(mm_mech, n_clients, len(cols), seed=seed)

    # mechanisms distribution across clients
    missing_mech_dist = generate_missing_mech_funcs(mm_dist, mm_funcs, n_clients, len(cols), seed=seed)

    ####################################################################################################################
    # configuration for each client
    ret = [{
        "missing_ratio": missing_ratios[i],
        "missing_mech_type": missing_mech_types[i],
        "missing_mech_dist": missing_mech_dist[i],
        "missing_features": cols,
    } for i in range(n_clients)]

    return ret


# if mm_strategy == "fixed":  # fixed@mr=0.1-mm=mcar-mfunc=lr-k=0.5
    #     mr = float(mm_params['mr'])
    #     missing_ratio = [[mr for _ in range(len(cols))] for _ in range(n_clients)]
    #     missing_mechanism_type = [mm_params['mm'] for _ in range(n_clients)]
    #
    #     if mm_params['mm'] == 'mcar':
    #         missing_mechanism_func = [[None for _ in range(len(cols))] for _ in range(n_clients)]
    #     else:
    #         if mm_params['mfunc'] == 'lr':
    #             mm_list = ['left', 'right']
    #         elif mm_params['mfunc'] == 'mt':
    #             mm_list = ['mid', 'tail']
    #         elif mm_params['mfunc'] == 'all':
    #             mm_list = ['left', 'right', 'mid', 'tail']
    #         else:
    #             raise ValueError(f'mm not found, params: {mm_params}')
    #
    #         k = float(mm_params['k'])
    #
    #         missing_mechanism_func = np.empty((n_clients, len(cols)), dtype='U20')
    #         np.random.seed(seed)
    #         for col in range(len(cols)):
    #             # randomly select ratio of clients
    #             client_idx = np.random.choice(np.arange(n_clients), int(k * n_clients), replace=False)
    #             # assign mechanism
    #             missing_mechanism_func[client_idx, col] = mm_list[0]
    #             mask = np.ones(n_clients, dtype=bool)
    #             mask[client_idx] = False
    #             missing_mechanism_func[mask, col] = mm_list[1]
    #
    # elif mm_strategy == "fixed2":  # fixed@mr=0.1-mm=mcar-mfunc=lr-k=0.5
    #     mr = float(mm_params['mr'])
    #     missing_ratio = [[mr for _ in range(len(cols))] for _ in range(n_clients)]
    #     missing_mechanism_type = [mm_params['mm'] for _ in range(n_clients)]
    #
    #     if mm_params['mm'] == 'mcar':
    #         missing_mechanism_func = [[None for _ in range(len(cols))] for _ in range(n_clients)]
    #     else:
    #         if mm_params['mfunc'] == 'lr':
    #             mm_list = ['left', 'right']
    #         elif mm_params['mfunc'] == 'mt':
    #             mm_list = ['mid', 'tail']
    #         elif mm_params['mfunc'] == 'all':
    #             mm_list = ['left', 'right', 'mid', 'tail']
    #         else:
    #             raise ValueError(f'mm not found, params: {mm_params}')
    #
    #         k = float(mm_params['k'])
    #
    #         missing_mechanism_func = np.empty((n_clients, len(cols)), dtype='U20')
    #         for col in range(len(cols)):
    #             # randomly select ratio of clients
    #             client_idx = np.arange(int(k * n_clients))
    #             # assign mechanism
    #             missing_mechanism_func[client_idx, col] = mm_list[0]
    #             mask = np.ones(n_clients, dtype=bool)
    #             mask[client_idx] = False
    #             missing_mechanism_func[mask, col] = mm_list[1]
    #
    # ####################################################################################################################
    # elif mm_strategy == "random":  # random@mrr=0.1-mrl=0.9-mm=mnar_quantile-mfunc=lr
    #     np.random.seed(seed)
    #     missing_ratio = np.random.uniform(
    #         float(mm_params['mrl']), float(mm_params['mrr']), (n_clients, len(cols))
    #     )
    #
    #     if mm_params['mm'] == 'mcar':
    #         missing_mechanism_func = [[None for _ in range(len(cols))] for _ in range(n_clients)]
    #     else:
    #         if mm_params['mfunc'] == 'lr':
    #             mm_list = ['left', 'right']
    #         elif mm_params['mfunc'] == 'mt':
    #             mm_list = ['mid', 'tail']
    #         elif mm_params['mfunc'] == 'all':
    #             mm_list = ['left', 'right', 'mid', 'tail']
    #         else:
    #             raise ValueError(f'mm not found, params: {mm_params}')
    #         missing_mechanism_func = np.random.choice(mm_list, (n_clients, len(cols)))
    #
    #     missing_mechanism_type = [[mm_params['mm'] for _ in range(len(cols))] for _ in range(n_clients)]
    #
    # ####################################################################################################################
    # elif mm_strategy == "random2":  # random@mrr=0.1-mrl=0.9-mm=mnar_quantile-mfunc=lr
    #     np.random.seed(seed)
    #     start = float(mm_params['mrl'])
    #     stop = float(mm_params['mrr'])
    #     step = int((stop - start) / 0.1 + 1)
    #     mr_list = np.linspace(start, stop, step, endpoint=True)
    #     missing_ratio = np.random.choice(mr_list, (n_clients, len(cols)))
    #
    #     if mm_params['mm'] == 'mcar':
    #         missing_mechanism_func = [[None for _ in range(len(cols))] for _ in range(n_clients)]
    #     else:
    #         if mm_params['mfunc'] == 'lr':
    #             mm_list = ['left', 'right']
    #         elif mm_params['mfunc'] == 'mt':
    #             mm_list = ['mid', 'tail']
    #         elif mm_params['mfunc'] == 'all':
    #             mm_list = ['left', 'right', 'mid', 'tail']
    #         else:
    #             raise ValueError(f'mm not found, params: {mm_params}')
    #         missing_mechanism_func = np.random.choice(mm_list, (n_clients, len(cols)))
    #
    #     missing_mechanism_type = [[mm_params['mm'] for _ in range(len(cols))] for _ in range(n_clients)]
    #
    # ####################################################################################################################
    # elif mm_strategy == "perfect_comp_lr":  # perfect_comp@mm=mnar_quantile
    #     mm_list = [(0.3, 0), (0.4, 0), (0.5, 0), (0.6, 0), (0.7, 0), (0.3, 1), (0.4, 1), (0.6, 1), (0.7, 1)]
    #     mm_func_list = ['left', 'right']
    #     missing_ratio = np.zeros((n_clients, len(cols)))
    #     missing_mechanism_func = np.empty((n_clients, len(cols)), dtype='U20')
    #     missing_mechanism_type = [[mm_params['mm'] for _ in range(len(cols))] for _ in range(n_clients)]
    #
    #     # sample
    #     np.random.seed(seed)
    #     cols_mechs = np.random.choice(np.arange(len(mm_list)), len(cols))
    #     client_idxs = np.random.choice(np.arange(n_clients), len(cols))
    #     for col in range(len(cols)):
    #         # randomly select a client
    #         client_idx = client_idxs[col]
    #         # assign missing ratio
    #         mr = mm_list[cols_mechs[col]][0]
    #         missing_ratio[client_idx, col] = mr
    #         missing_ratio[np.arange(missing_ratio.shape[0]) != client_idx, col] = 1 - mr
    #         # assign mechanism
    #         mm = int(mm_list[cols_mechs[col]][1])
    #         missing_mechanism_func[client_idx, col] = mm_func_list[mm]
    #         missing_mechanism_func[np.arange(missing_mechanism_func.shape[0]) != client_idx, col] = mm_func_list[1 - mm]
    #
    # ####################################################################################################################
    # elif mm_strategy == "imperfect_comp_lr":  # random@mrr=0.1-mrl=0.9-mm=mnar_quantile-mfunc=lr
    #
    #     mm_list = [(0.3, 0), (0.4, 0), (0.5, 0), (0.6, 0), (0.7, 0), (0.3, 1), (0.4, 1), (0.6, 1), (0.7, 1)]
    #     mechs = ['left', 'right']
    #     missing_ratio = np.zeros((n_clients, len(cols)))
    #     missing_mechanism_func = np.empty((n_clients, len(cols)), dtype='U20')
    #     missing_mechanism_type = [[mm_params['mm'] for _ in range(len(cols))] for _ in range(n_clients)]
    #
    #     np.random.seed(seed)
    #     cols_mechs = np.random.choice(np.arange(len(mm_list)), len(cols))
    #     client_idxs = np.random.choice(np.arange(n_clients), len(cols))
    #     for col in range(len(cols)):
    #         left_mech = [(0.3, 0), (0.4, 0), (0.5, 0), (0.6, 0), (0.7, 0)]
    #         right_mech = [(0.3, 1), (0.4, 1), (0.5, 1), (0.6, 1), (0.7, 1)]
    #         mm = int(mm_list[cols_mechs[col]][1])
    #         mr = mm_list[cols_mechs[col]][0]
    #         if mm == 0:
    #             imperfect_mechs = [item for item in right_mech if not math.isclose(item[0], 1 - mr, rel_tol=1e-9)]
    #         else:
    #             imperfect_mechs = [item for item in left_mech if not math.isclose(item[0], 1 - mr, rel_tol=1e-9)]
    #         random.seed(seed + col)
    #         imperfect_mech = random.sample(imperfect_mechs, 1)[0]
    #         mm2 = int(imperfect_mech[1])
    #         mr2 = imperfect_mech[0]
    #
    #         # randomly select a client
    #         client_idx = client_idxs[col]
    #         # assign missing ratio
    #         missing_ratio[client_idx, col] = mr
    #         missing_ratio[np.arange(missing_ratio.shape[0]) != client_idx, col] = mr2
    #
    #         # assign mechanism
    #         missing_mechanism_func[client_idx, col] = mechs[mm]
    #         missing_mechanism_func[np.arange(missing_mechanism_func.shape[0]) != client_idx, col] = mechs[mm2]
    #
    # ####################################################################################################################
    # else:
    #     raise NotImplementedError(f'mm_strategy not found, params: {mm_strategy}')
