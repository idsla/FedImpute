import numpy as np
from modules.sampling import dirichlet_noniid_partition, generate_dirichlet_noniid_distribution

np.random.seed(42)


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


if __name__ == '__main__':
    import pandas as pd

    ret = dirichlet_noniid_partition(
        alphas=[1, 1, 1, 1, 1],
        data=pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
    )
    print(ret)

    ret = generate_dirichlet_noniid_distribution(alphas=[1, 1, 1, 1, 1])
    print(ret)
