import numpy as np


def dirichlet(features, n_clients=10):
    features_groups = np.array_split(np.array(features), indices_or_sections=4)
    clients_groups = np.array_split(np.arange(0, n_clients), indices_or_sections=4)
    np.random.seed(0)
    ret = np.zeros((n_clients, len(features)))
    for clients, feature_set in zip(clients_groups, features_groups):
        print(clients, feature_set)
        num_clients = len(clients)
        num_features = len(feature_set)
        partitions = np.random.dirichlet([1] * num_clients, num_features).transpose()
        for client_idx, client in enumerate(clients):
            client_ret = np.zeros(len(features)) + 0.05 + 0.1*np.random.random()
            client_ret[feature_set] = partitions[client_idx]
            ret[client] = client_ret

    ret2 = []
    for i in range(n_clients):
        print(i)
        ret2.append(
            {
                idx: ret[i, idx] for idx in range(len(ret[i]))
            }
        )

    return ret2


if __name__ == '__main__':
    dirichlet([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
