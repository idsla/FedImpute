from sklearn.manifold import TSNE
import gower
import numpy as np


def eval_tsne(origin_datas, imputed_datas):
    origin_data = np.concatenate(origin_datas, axis=0)
    imputed_data = np.concatenate(imputed_datas, axis=0)

    # overall
    plot_data = np.concatenate((origin_data, imputed_data), axis=0)
    N1 = origin_data.shape[0]
    N2 = imputed_data.shape[0]
    colors = ["red" for i in range(N1)] + ["blue" for i in range(N2)]
    tsne = TSNE(
        metric='precomputed', n_components=2, verbose=0, n_iter=1000, perplexity=40,
        n_iter_without_progress=300, init='random', n_jobs=-1, random_state=42
    )

    tsne_results = tsne.fit_transform(np.clip(gower.gower_matrix(plot_data), 0, 1))

    return tsne_results, colors, N1, N2

    # client specific
    # for client_index, client_label in zip(client_indices, client_labels):
    #     # data for one client
    #     origin_data = origin_datas[client_index]
    #     imputed_data = imputed_datas[client_index]
    #     plot_data = np.concatenate((origin_data, imputed_data), axis=0)
    #
    #     # Parameters
    #     N1 = origin_data.shape[0]
    #     N2 = imputed_data.shape[0]
    #     colors = ["red" for i in range(N1)] + ["blue" for i in range(N2)]
    #
    #     # TSNE anlaysis
    #     tsne = TSNE(metric='precomputed', n_components=2, verbose=0, n_iter=1000, perplexity=40,
    #                 n_iter_without_progress=300, init='random')
    #     tsne_results = tsne.fit_transform(np.clip(gower.gower_matrix(plot_data), 0, 1))
    #     print('Ploting TSNE for client {} with method {} ...'.format(client_label, method))
    #     tsne_results_dict[(client_label, method)] = tsne_results
    #     colors_result[(client_label, method)] = colors


def plot_tsne():
    pass