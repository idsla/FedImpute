import pandas as pd
import numpy as np
import random

def generate_alphas(alpha, n):
    return [alpha for _ in range(n)]


def dirichlet_noniid_partition(alphas, data):
    np.random.seed(21)
    random.seed(21)
    partition_distribution = np.random.dirichlet(alphas, 1)
    indices = data.index
    partition_split_point = (np.cumsum(partition_distribution)[:-1] * len(data)).astype(int)
    partition_indices = np.split(indices, partition_split_point)
    client_idcs = partition_indices
    #print(client_idcs)

    return client_idcs


def generate_dirichlet_noniid_distribution(alphas):
    return np.random.dirichlet(alphas, 1)


#################################################################################################################
# MISC
#################################################################################################################
def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)

    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    client_idcs = [[] for _ in range(n_clients)]

    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs