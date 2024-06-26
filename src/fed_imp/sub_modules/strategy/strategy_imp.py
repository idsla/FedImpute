import numpy as np
from copy import deepcopy

from .fedavg import fedavg, fedavgs, fedavgh, fedavg2, fedavgcross, testavg, fedavg_vae
from .fedmech import fedmechclw, fedmechclwcl, fedmechcl2, fedmechw, fedmechcl4, fedmechw_new, fedmechw_new2
from .fedwavg import fedwavg, fedwavg2, fedwavg3
from .fedwavgcl import fedwavgcl


class StrategyImputation:
    """
    Strategy for aggregation weights of federated imputation model
    """

    def __init__(
            self,
            strategy="local",
            params: dict = None,
    ) -> None:

        # aggregation strategy
        self.strategy = strategy
        if strategy == 'local':
            self.initial_strategy = 'local'
        elif strategy == 'local_vae':
            self.initial_strategy = 'local'
        elif strategy == 'local_gain':
            self.initial_strategy = 'local'
        elif strategy == 'central':
            self.initial_strategy = 'local'
        elif strategy == 'central2':
            self.initial_strategy = 'central2'
        elif strategy == 'central_vae':
            self.initial_strategy = 'central2'
        elif strategy == 'central_gain':
            self.initial_strategy = 'central2'
        elif strategy.startswith('fedavg'):
            self.initial_strategy = 'fedavg'
        elif strategy == 'testavg':
            print('testavg')
            self.initial_strategy = 'testavg'
        elif strategy == 'testavg2':
            print('testavg2')
            self.initial_strategy = 'fedavg'
        elif strategy == 'testavg3':
            print('testavg3')
            self.initial_strategy = 'fedavg'
        elif strategy.startswith('fedwavg') and not strategy.startswith('fedwavgcl'):
            self.initial_strategy = 'fedavg'
        elif strategy.startswith('fedwavgcl'):
            self.initial_strategy = 'fedavg'
        elif strategy.startswith('fedmech'):
            self.initial_strategy = 'fedavg'
        elif strategy.startswith('cafe'):
            self.initial_strategy = 'fedavg'
        else:
            raise ValueError(
                'Unknown imputation model aggregation strategy: {}'.format(strategy)
            )

        # parameters for aggregation
        self.alpha = params.get('alpha', 0.33)
        self.beta = params.get('beta', 0.33)
        self.params = params

    def aggregate(
            self, weights, losses, missing_infos, client_groups, project_matrix, ms_coefs, top_k_idx_clients=None,
            round=None):

        w = None
        if self.strategy == 'local':
            agg_weight = None
        elif self.strategy == 'central':
            agg_weight = None
        elif self.strategy == 'central2':
            clients_weights = np.array(list(weights.values()))
            agg_weight = clients_weights[-1, :]
        # ==============================================================================================================
        # VAE GAN Algorithm
        # ==============================================================================================================
        elif self.strategy == 'fedavg_vae':
            agg_weight, w = fedavg_vae(weights, missing_infos)
        elif self.strategy == 'central_vae':
            client_weights = list(weights.values())
            agg_weight = deepcopy(client_weights[-1])
        elif self.strategy == 'fedavg_gain':
            agg_weight, w = fedavg_vae(weights, missing_infos)
        elif self.strategy == 'central_gain':
            client_weights = list(weights.values())
            agg_weight = deepcopy(client_weights[-1])
        # ==============================================================================================================
        # Average Algorithm
        # ==============================================================================================================
        elif self.strategy == 'fedavg':
            agg_weight = fedavg(weights)
        elif self.strategy == 'fedavg-s':
            agg_weight, w = fedavgs(weights, missing_infos)
        # ==============================================================================================================
        # Missing Mechanism Average Algorithm
        # ==============================================================================================================
        elif self.strategy == 'fedmechw':
            agg_weight, w = fedmechw(weights, missing_infos, ms_coefs, self.params)
        elif self.strategy.startswith('fedmechw_new'):
            algo_params = self.strategy.split('-')
            if len(algo_params) == 1:
                params = self.params
            else:
                params = {'alpha': 0.95, 'gamma': 0.05, 'client_thres': 1.0, 'scale_factor': 4}
                param_dict = process_algo_params_str(algo_params[1])
                params['alpha'] = param_dict['a'] if 'a' in param_dict else params['alpha']
                params['gamma'] = param_dict['g'] if 'g' in param_dict else params['gamma']
                params['scale_factor'] = param_dict['sf'] if 'sf' in param_dict else params['scale_factor']
            #print(params)
            agg_weight, w = fedmechw_new(weights, missing_infos, ms_coefs, params, round=round)
        elif self.strategy.startswith('cafe'):
            params = self.params
            agg_weight, w = fedmechw_new(weights, missing_infos, ms_coefs, params, round=round)
        # ==============================================================================================================
        else:
            raise ValueError(
                'Unknown imputation model aggregation strategy: {}'.format(self.strategy)
            )

        return agg_weight, w

    def aggregate_initial(self, values, sample_sizes, missing_ratios, client_groups_dict):

        if self.initial_strategy == 'local':
            agg_value = None

        elif self.initial_strategy == 'central2':
            values = np.array(values)
            agg_value = values[-1, :]

        elif self.initial_strategy == 'fedavg':
            values = np.array(values)
            agg_value = np.mean(values, axis=0)

        elif self.initial_strategy == 'testavg':
            values = np.array(values)
            n_clients = len(values)
            cluster1_idx = list(range(0, int(n_clients * 0.1)))
            cluster2_idx = list(range(int(n_clients * 0.1), n_clients))
            cluster1_values = values[cluster1_idx]
            cluster2_values = values[cluster2_idx]
            ret = {}
            for i in range(n_clients):
                if i in cluster1_idx:
                    ret[i] = np.mean(cluster2_values, axis=0)
                else:
                    ret[i] = np.mean(cluster1_values, axis=0)
            return ret

        elif self.initial_strategy == 'fedwavg':
            values = np.array(values)
            missing_ratios = 1 - np.array(missing_ratios) + 0.0001
            sample_sizes = np.array(sample_sizes)
            sample_sizes = sample_sizes / np.sum(sample_sizes, axis=0)
            missing_ratios = missing_ratios / missing_ratios.sum(axis=0)
            scale_factor = 2
            weights = (0.9 * sample_sizes + 0.1 * missing_ratios) ** scale_factor
            agg_value = np.average(values, axis=0, weights=weights)

        elif self.initial_strategy == 'fedwavgcl':
            values = np.array(values)
            missing_ratios = 1 - np.array(missing_ratios) + 0.0001
            agg_value = np.zeros(values.shape[1])
            sample_sizes = np.array(sample_sizes)
            scale_factor = 2

            for col in range(values.shape[1]):
                groups = client_groups_dict[col]
                n_clusters = len(groups)
                weights_clusters = []
                ms_ratio_clusters = []
                ms_weight_clusters = []
                for i in range(n_clusters):
                    if len(groups[i]) == 0:
                        continue
                    ms_weights_w = missing_ratios[:, col][groups[i]] ** scale_factor
                    ms_weights_w = ms_weights_w / ms_weights_w.sum()
                    # average in-group weights using losses
                    weights_group = values[:, col][groups[i]]
                    weight_average = np.average(weights_group, axis=0, weights=ms_weights_w)
                    weights_clusters.append(weight_average)
                    # average ms_weights
                    ms_ratio_group = sample_sizes[:, col][groups[i]]
                    ms_ratio_clusters.append(ms_ratio_group.mean())
                    ms_weights = np.average(missing_ratios[:, col][groups[i]], weights=ms_weights_w)
                    ms_weight_clusters.append(ms_weights)

                weights_avg_clusters = np.array(weights_clusters)
                ms_ratio_avg_clusters = np.array(ms_ratio_clusters)
                ms_weight_clusters = np.array(ms_weight_clusters)

                ms_ratio_avg_clusters = ms_ratio_avg_clusters / ms_ratio_avg_clusters.sum()
                ms_weight_clusters = ms_weight_clusters / ms_weight_clusters.sum()
                final_weights = (0.9 * ms_ratio_avg_clusters + 0.1 * ms_weight_clusters) ** scale_factor

                # take average across clusters
                agg_value[col] = np.average(weights_avg_clusters, axis=0, weights=final_weights)
        else:
            raise ValueError(
                'Unknown imputation model aggregation strategy: {}'.format(self.strategy)
            )

        return agg_value


def process_algo_params_str(algo_param_str):
    # Split the string into key-value pairs based on '_'
    pairs = algo_param_str.split('_')

    # Split each pair into key and value based on '=' and convert to appropriate type
    param_dict = {}
    for pair in pairs:
        key, value = pair.split('=')
        # Attempt to convert to integer
        if value.isdigit():
            value = int(value)
        # Attempt to convert to float
        elif value.replace('.', '', 1).isdigit():
            value = float(value)
        # Leave as string if neither integer nor float
        param_dict[key] = value

    return param_dict
