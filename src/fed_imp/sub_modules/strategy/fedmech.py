from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from config import settings


def fedmodelw(weights, missing_infos, ms_coefs, params):

    scale_factor = settings['algo_params']['scale_factor']
    weights = np.array(list(weights.values()))
    # number of client considered
    client_thres = int(len(weights) * params['client_thres'])

    # samples sizes
    sample_sizes = np.array([v['sample_row_pct'] + 0.0001 for v in missing_infos.values()])
    missing_pct = np.array([(1 - v['missing_cell_pct']) + 0.0001 for v in missing_infos.values()])
    model_sim_dist = mech_cos_sim_matrix(weights)

    alpha = 1
    beta = 0

    final_parameters, w = [], []
    for client_idx in range(len(weights)):
        # select top-k client whose mech_sim_dist is larger
        top_k_idx = np.argsort(model_sim_dist[client_idx])[-client_thres:]

        # adjust weights
        mech_sim_w = (model_sim_dist[client_idx][top_k_idx] + 0.00001) ** scale_factor
        mech_sim_w = mech_sim_w / np.sum(mech_sim_w)

        sample_size_w = (sample_sizes[top_k_idx]) ** scale_factor
        sample_size_w = sample_size_w / np.sum(sample_size_w)

        missing_pct_w = (missing_pct[top_k_idx]) ** scale_factor
        missing_pct_w = missing_pct_w / np.sum(missing_pct_w)

        final_w = (alpha * (mech_sim_w) + beta * (sample_size_w) + (1 - alpha - beta) * (missing_pct_w)) ** scale_factor

        # average parameters
        avg_parameters = np.average(weights[top_k_idx], axis=0, weights=final_w)
        final_parameters.append(avg_parameters)
        w.append(final_w / np.sum(final_w))

    return final_parameters, w


def fedmechclw(weights, missing_infos, ms_coefs, params):
    #scale_factor = settings['algo_params']['scale_factor']

    # parameters of every model
    weights = np.array(list(weights.values()))

    # samples sizes
    sample_sizes = np.array([v['sample_row_pct'] + 0.0001 for v in missing_infos.values()])
    missing_pct = np.array([(1 - v['missing_cell_pct']) + 0.0001 for v in missing_infos.values()])

    # ms_coefs
    ms_coefs = np.array(list(ms_coefs.values()))
    mech_sim_dist = mech_cos_sim_matrix(ms_coefs)

    cluster_thres = params['thres1']
    alpha = params['alpha']
    scale_factor = params['scale_factor']

    ################################################################################################
    # Cluster clients based on cosine similarity of missing mechanisms
    ################################################################################################
    # cluster clients based on cosine similarity
    groups, centroids = clustering(ms_coefs, threshold=cluster_thres)

    ################################################################################################
    # In Group Weight Averaging parameters based on sample sizes
    ################################################################################################
    group_avg_parameters = []
    for i in range(len(groups)):

        # all sample_sizes and weights in the same group
        client_indices = groups[i]
        w1 = sample_sizes[client_indices] ** scale_factor
        w1 = w1 / np.sum(w1)
        w2 = missing_pct[client_indices] ** scale_factor
        w2 = w2 / np.sum(w2)
        w = (alpha * w1 + (1 - alpha) * w2) ** scale_factor
        parameters = weights[client_indices]

        # average parameters
        avg_parameters = np.average(parameters, axis=0, weights=w)
        group_avg_parameters.append(avg_parameters)

    ################################################################################################
    # Each Client Cross Group Weight Averaging parameters based on distance between groups
    ################################################################################################
    final_parameters, w = [], []
    client_thres = int(len(weights) * params['client_thres'])
    for client in range(len(ms_coefs)):

        # select top-k dissimilar clients based on mech sim weight and get filtered groups
        top_k_idx = np.argsort(mech_sim_dist[client])[-client_thres:]
        filtered_centroids = []
        filtered_group_avg_parameters = []
        for group, centroid, group_avg_parameter in zip(groups, centroids, group_avg_parameters):
            for idx in top_k_idx:
                if idx in group:
                    filtered_centroids.append(centroid)
                    filtered_group_avg_parameters.append(group_avg_parameter)
                    break

        # distance between client and each group
        distances = mech_cos_sim_distance(ms_coefs[client], filtered_centroids)

        # weighted average based on distance
        distances = (distances + 0.00001) ** scale_factor

        # final parameters
        parameters = np.average(filtered_group_avg_parameters, axis=0, weights=distances)
        w.append(distances/np.sum(distances))
        final_parameters.append(parameters)

    return final_parameters, w


def fedmechw(weights, missing_infos, ms_coefs, params, sigmoid = False, filter_sim_mm = False, filter_sim_lm = False):
    '''Three factors Weighted Average'''

    #scale_factor = settings['algo_params']['scale_factor']
    scale_factor = params['scale_factor']
    alpha = params['alpha']

    # parameters of every model
    weights = np.array(list(weights.values()))

    # number of client considered
    client_thres = int(len(weights) * params['client_thres'])

    # samples sizes
    sample_sizes = np.array([v['sample_row_pct'] + 0.0001 for v in missing_infos.values()])
    missing_pct = np.array([(1 - v['missing_cell_pct']) + 0.0001 for v in missing_infos.values()])

    ms_coefs = np.array(list(ms_coefs.values()))
    mech_sim_dist = mech_cos_sim_matrix(ms_coefs)
    model_sim_dist = mech_cos_sim_matrix(weights)

    if sigmoid:
        mech_sim_dist = 1 - 1 / (1 + np.exp(10 * (mech_sim_dist - 0.5)))

    if filter_sim_mm:
        mech_sim_dist = np.where(mech_sim_dist > 0.3, mech_sim_dist, 0.001)
    
    if filter_sim_lm:
        mech_sim_dist = np.where(model_sim_dist > 0.05, mech_sim_dist, 0.001)

    final_parameters, w = [], []
    for client_idx in range(len(weights)):
        # select top-k client whose mech_sim_dist is larger
        top_k_idx = np.argsort(mech_sim_dist[client_idx])[-client_thres:]

        # adjust weights
        mech_sim_w = (mech_sim_dist[client_idx][top_k_idx] + 0.00001)
        #mech_sim_w = mech_sim_w / np.sum(mech_sim_w)

        # normalized sample sizes weights - same as simple average
        sample_size_w = (sample_sizes[top_k_idx])
        sample_size_w = sample_size_w / np.max(sample_size_w)

        # missing pct weights
        # missing_pct_w = (missing_pct[top_k_idx]) ** scale_factor
        # missing_pct_w = missing_pct_w / np.sum(missing_pct_w)

        final_w = (alpha * (mech_sim_w) + (1 - alpha) * (sample_size_w))**scale_factor #(1 - alpha - beta) * (missing_pct_w))** scale_factor

        # average parameters
        avg_parameters = np.average(weights[top_k_idx], axis=0, weights=final_w)
        final_parameters.append(avg_parameters)
        w.append(final_w/np.sum(final_w))

    return final_parameters, w


def fedmechw_new(weights, missing_infos, ms_coefs, params, sigmoid = False, filter_sim_mm = False, filter_sim_lm = False, round = None):
    '''Three factors Weighted Average'''

    #scale_factor = settings['algo_params']['scale_factor']
    scale_factor = params['scale_factor']
    alpha = params['alpha']
    gamma = params['gamma']

    # parameters of every model
    weights = np.array(list(weights.values()))

    # number of client considered
    client_thres = int(len(weights) * params['client_thres'])

    # samples sizes
    sample_sizes = np.array([v['sample_size'] + 1e-4 for v in missing_infos.values()])
    # missing_pct = np.array([(1 - v['missing_cell_pct']) + 0.0001 for v in missing_infos.values()])

    ms_coefs = np.array(list(ms_coefs.values()))
    mech_sim_dist = mech_cos_sim_matrix(ms_coefs)
    model_sim_dist = mech_cos_sim_matrix(weights)

    if sigmoid:
        mech_sim_dist = 1 - 1 / (1 + np.exp(10 * (mech_sim_dist - 0.5)))

    if filter_sim_mm:
        mech_sim_dist = np.where(mech_sim_dist > 0.05, mech_sim_dist, 0.001)
    
    if filter_sim_lm:
        mech_sim_dist = np.where(model_sim_dist > 0.05, mech_sim_dist, 0.001)

    final_parameters, w = [], []
    stats_compl, stats_idx, stats_sampl = [], [], []
    for client_idx in range(len(weights)):
        
        other_clients_idx = [i for i in range(len(weights)) if i != client_idx]
        client_thres = int(len(other_clients_idx) * params['client_thres'])
        
        # select top-k client whose mech_sim_dist is larger
        sorted_idx = np.argsort(mech_sim_dist[client_idx])
        sorted_idx = np.array([i for i in sorted_idx if i != client_idx])
        top_k_idx = sorted_idx[-client_thres:]

        # adjust weights
        mech_sim_w = (mech_sim_dist[client_idx][top_k_idx] + 0.00001)
        #mech_sim_w = mech_sim_w / np.sum(mech_sim_w)

        # normalized sample sizes weights - same as simple average
        sample_size_w = (sample_sizes[top_k_idx])
        sample_size_w = sample_size_w / np.max(sample_sizes)

        # missing pct weights
        # missing_pct_w = (missing_pct[top_k_idx]) ** scale_factor
        # missing_pct_w = missing_pct_w / np.sum(missing_pct_w)
        stats_compl.append(mech_sim_w)
        stats_idx.append(top_k_idx)
        stats_sampl.append(sample_size_w)

        final_w = (alpha * (mech_sim_w) + (1 - alpha) * (sample_size_w))**scale_factor #(1 - alpha - beta) * (missing_pct_w))** scale_factor
        # average parameters of others
        other_avg_parameters = np.average(weights[top_k_idx], axis=0, weights=final_w)
        
        # weighted average between self and others
        avg_parameters = gamma * weights[client_idx] + (1 - gamma) * other_avg_parameters
        
        final_parameters.append(avg_parameters)
        w.append(final_w/np.sum(final_w))
        
    # if round == 19:
    #     print('====================================================')
    #     for row in range(len(stats_compl)):
    #         print("idx: {}".format(stats_idx[row]))
    #         print("cmp: {}".format(stats_compl[row]))
    #         print("smp: {}".format(stats_sampl[row]))
    #     print(sample_sizes)
        
    return final_parameters, w


def fedmechw_new2(weights, missing_infos, ms_coefs, params, sigmoid = False, filter_sim_mm = False, filter_sim_lm = False, round = None):
    '''Three factors Weighted Average'''

    #scale_factor = settings['algo_params']['scale_factor']
    scale_factor = params['scale_factor']
    alpha = params['alpha']
    gamma = params['gamma']
    mm_thres = params['mm_thres']
    # sample_size_threshold = params['s_thres']
    # s_steep = params['s_steep']

    # parameters of every model
    weights = np.array(list(weights.values()))

    # number of client considered
    client_thres = int(len(weights) * params['client_thres'])

    # samples sizes
    sample_sizes = np.array([v['sample_row_pct'] + 0.0001 for v in missing_infos.values()])
    # missing_pct = np.array([(1 - v['missing_cell_pct']) + 0.0001 for v in missing_infos.values()])

    ms_coefs = np.array(list(ms_coefs.values()))
    mech_sim_dist = mech_cos_sim_matrix(ms_coefs)
    model_sim_dist = mech_cos_sim_matrix(weights)

    if sigmoid:
        mech_sim_dist = 1 - 1 / (1 + np.exp(10 * (mech_sim_dist - 0.5)))

    if filter_sim_mm:
        mech_sim_dist = np.where(mech_sim_dist > 0.05, mech_sim_dist, 0.001)
    
    if filter_sim_lm:
        mech_sim_dist = np.where(model_sim_dist > 0.05, mech_sim_dist, 0.001)

    final_parameters, w = [], []
    for client_idx in range(len(weights)):
        
        other_clients_idx = [i for i in range(len(weights)) if i != client_idx]
        client_thres = int(len(other_clients_idx) * params['client_thres'])
        
        # select top-k client whose mech_sim_dist is larger
        sorted_idx = np.argsort(mech_sim_dist[client_idx])
        sorted_idx = np.array([i for i in sorted_idx if i != client_idx])
        top_k_idx = sorted_idx[-client_thres:]

        # adjust weights
        mech_sim_w = (mech_sim_dist[client_idx][top_k_idx] + 0.00001)
        #mech_sim_w = mech_sim_w / np.sum(mech_sim_w)

        # normalized sample sizes weights - same as simple average
        sample_size_w = (sample_sizes[top_k_idx])
        sample_size_w = sample_size_w / np.max(sample_size_w)
        # sample_size_w = 1 - 1 / (1 + np.exp(s_steep * (sample_size_w - sample_size_threshold)))

        # missing pct weights
        # missing_pct_w = (missing_pct[top_k_idx]) ** scale_factor
        # missing_pct_w = missing_pct_w / np.sum(missing_pct_w)
        
        if round >= 19:
            if mech_sim_w.max() <= mm_thres:
                alpha = 0
            else:
                alpha = 1

        final_w = (alpha * (mech_sim_w) + (1 - alpha) * (sample_size_w))**scale_factor #(1 - alpha - beta) * (missing_pct_w))** scale_factor
        # average parameters of others
        other_avg_parameters = np.average(weights[top_k_idx], axis=0, weights=final_w)
        
        # weighted average between self and others
        avg_parameters = gamma * weights[client_idx] + (1 - gamma) * other_avg_parameters
        
        final_parameters.append(avg_parameters)
        w.append(final_w/np.sum(final_w))

    return final_parameters, w


def fedmechclwcl(weights, missing_infos, ms_coefs):
    scale_factor = settings['algo_params']['scale_factor']

    # parameters of every model
    weights = np.array(list(weights.values()))

    # samples sizes
    sample_sizes = np.array([v['sample_row_pct'] + 0.0001 for v in missing_infos.values()])
    missing_pct = np.array([(1 - v['missing_cell_pct']) + 0.0001 for v in missing_infos.values()])

    # ms_coefs
    ms_coefs = np.array(list(ms_coefs.values()))
    mech_sim_dist = mech_cos_sim_matrix(ms_coefs)

    cluster_thres = settings['algo_params']['fedmechclwcl']['thres1']
    wcl_thres = settings['algo_params']['fedmechclwcl']['thres2']

    ################################################################################################
    # Cluster clients based on cosine similarity of missing mechanisms
    ################################################################################################
    # cluster clients based on cosine similarity
    groups, centroids = clustering(ms_coefs, threshold=cluster_thres)

    ################################################################################################
    # In Group Weight Averaging parameters based on sample sizes
    ################################################################################################
    group_avg_parameters = []
    for i in range(len(groups)):
        # all sample_sizes and weights in the same group
        client_indices = groups[i]

        # clustering clients within each group
        w1 = sample_sizes[client_indices]
        groups2 = clustering1(
            w1, client_indices, threshold=wcl_thres)

        weights_avgs, cross_w = [], []
        for i in range(len(groups2)):
            if len(groups2[i]) == 0:
                continue

            # compute in-group weights
            w = (missing_pct[groups2[i]] + 1e-4) ** scale_factor
            w = w / w.sum()

            # average in-group weights using losses
            weights_avgs.append(
                np.average(weights[groups2[i]], axis=0, weights=w)
            )

            # average cross group weights
            cross_w.append(np.average(sample_sizes[groups2[i]], weights=w))

        weights_avgs = np.array(weights_avgs)
        cross_w = np.array(cross_w) ** scale_factor

        # take average across clusters
        avg_parameters = np.average(weights_avgs, axis=0, weights=cross_w)
        group_avg_parameters.append(avg_parameters)

    ################################################################################################
    # Each Client Cross Group Weight Averaging parameters based on distance between groups
    ################################################################################################
    final_parameters = []
    client_thres = int(len(weights) * settings['algo_params']['fedmechclwcl']['client_thres'])
    for client in range(len(ms_coefs)):

        # select top-k dissimilar clients based on mech sim weight and get filtered groups
        top_k_idx = np.argsort(mech_sim_dist[client])[-client_thres:]
        filtered_centroids = []
        filtered_group_avg_parameters = []
        for group, centroid, group_avg_parameter in zip(groups, centroids, group_avg_parameters):
            for idx in top_k_idx:
                if idx in group:
                    filtered_centroids.append(centroid)
                    filtered_group_avg_parameters.append(group_avg_parameter)
                    break

        # distance between client and each group
        distances = mech_cos_sim_distance(ms_coefs[client], filtered_centroids)

        # weighted average based on distance
        distances = (distances + 0.00001) ** scale_factor

        # final parameters
        parameters = np.average(filtered_group_avg_parameters, axis=0, weights=distances)
        final_parameters.append(parameters)

    return final_parameters


def fedmechcl2(weights, missing_infos, ms_coefs, top_k_idx_clients=None):
    """Three Levels Clustering"""
    scale_factor = settings['algo_params']['scale_factor']
    # parameters of every model
    weights = np.array(list(weights.values()))

    # missing ratio pct
    non_missing_pcts = np.array([v['sample_row_pct'] + 0.0001 for v in missing_infos.values()])

    # missing cell pct in covariate
    ms_cell_weights = np.array([1 - v['missing_cell_pct'] + 0.0001 for v in missing_infos.values()])

    # number of client considered
    client_thres = int(len(weights) * settings['algo_params']['fedmechcl']['client_thres'])

    # missing mechanism similarity
    ms_coefs = np.array(list(ms_coefs.values()))
    mech_sim_dist = mech_cos_sim_matrix(ms_coefs)

    ################################################################################################
    # Calculate parameters for each client using two level clustering
    ################################################################################################
    final_parameters = []
    top_k_idx_clients_new = {}
    for client_idx in range(len(weights)):
        # select top-k client whose mech_sim_dist is larger
        if top_k_idx_clients is None:
            top_k_idx = np.argsort(mech_sim_dist[client_idx])[-client_thres:]
            top_k_idx_clients_new[client_idx] = top_k_idx
        else:
            top_k_idx = top_k_idx_clients[client_idx]

        # first level clustering based on weights
        alpha = settings['algo_params']['fedmechcl']['alpha']
        beta = settings['algo_params']['fedmechcl']['beta']
        w1 = (mech_sim_dist[client_idx][top_k_idx] + 0.00001)
        first_level_groups = clustering1(
            w1, top_k_idx, threshold=settings['algo_params']['fedmechcl']['thres1'])

        first_cross_group_weights, first_group_weights = [], []
        for group in first_level_groups:
            # second level clustering based on non_missing_pct
            # second_level_groups = clustering1(non_missing_pcts[group] + 0.00001, group,
            #                                   threshold=settings['algo_params']['fedmechcl']['thres2'])
            #
            # second_cross_group_weights, second_group_weights = [], []
            # for second_group in second_level_groups:
            #     #     # second level aggregation -> within group: missing cell pct, cross group: non_missing_pct
            #     second_cross_group_weights.append(non_missing_pcts[second_group].mean())
            #
            #     second_group_weights.append(
            #         np.average(weights[second_group], axis=0,
            #                    weights=(ms_cell_weights[second_group] + 1e-4) ** scale_factor)
            #     )
            #
            # second_cross_group_weights = np.array(second_cross_group_weights)
            # second_group_weights = np.array(second_group_weights)
            #
            # # final weights for first level group
            # second_final_weights = np.average(second_group_weights, axis=0,
            #                                   weights=second_cross_group_weights ** scale_factor)

            # first level aggregation -> cross group: missing_mech_sim
            w2 = (non_missing_pcts[group] + 0.00001)**scale_factor
            w2 = w2 / np.sum(w2)
            w3 = (ms_cell_weights[group] + 0.00001)**scale_factor
            w3 = w3 / np.sum(w3)

            w = beta * w2 + (1 - beta) * w3

            first_cross_group_weights.append((mech_sim_dist[client_idx][group] + 0.00001).mean())
           
            first_group_weights.append(
                np.average(weights[group], axis=0, weights=(w + 1e-4) ** scale_factor)
            )
            # first_group_weights.append(second_final_weights)
            # first_group_weights.append(
            #     np.average(weights[group], axis=0, weights=(mech_sim_dist[client_idx][group] + 1e-4) ** scale_factor))

        first_cross_group_weights = np.array(first_cross_group_weights)
        first_group_weights = np.array(first_group_weights)

        final_w = np.average(first_group_weights, axis=0, weights=first_cross_group_weights ** scale_factor)
        final_parameters.append(final_w)

    return final_parameters, top_k_idx_clients_new


def fedmechcl4(weights, missing_infos, ms_coefs):
    """Three Levels Clustering"""
    scale_factor = settings['algo_params']['scale_factor']
    # parameters of every model
    weights = np.array(list(weights.values()))

    # missing ratio pct
    non_missing_pcts = np.array([v['sample_row_pct'] + 0.0001 for v in missing_infos.values()])

    # missing cell pct in covariate
    ms_cell_weights = np.array([1 - v['missing_cell_pct'] + 0.0001 for v in missing_infos.values()])

    # number of client considered
    client_thres = int(len(weights) * settings['algo_params']['fedmechcl2']['client_thres'])

    # missing mechanism similarity
    ms_coefs = np.array(list(ms_coefs.values()))
    mech_sim_dist = mech_cos_sim_matrix(ms_coefs)

    ################################################################################################
    # Calculate parameters for each client using two level clustering
    ################################################################################################
    final_parameters = []
    for client_idx in range(len(weights)):
        # select top-k client whose mech_sim_dist is larger
        top_k_idx = np.argsort(mech_sim_dist[client_idx])[-client_thres:]

        # first level clustering based on weights
        w1 = (mech_sim_dist[client_idx][top_k_idx] + 0.00001)
        first_level_groups = clustering1(
            w1, top_k_idx, threshold=settings['algo_params']['fedmechcl2']['thres1'])

        first_cross_group_weights, first_group_weights = [], []
        for group in first_level_groups:
            
            # second level clustering based on non_missing_pct
            second_level_groups = clustering1(non_missing_pcts[group] + 0.00001, group,
                                               threshold=settings['algo_params']['fedmechcl2']['thres2'])
            
            second_cross_group_weights, second_group_weights = [], []
            for second_group in second_level_groups:
                # second level aggregation -> within group: missing cell pct, cross group: non_missing_pct
                second_cross_group_weights.append(
                    np.average(
                        non_missing_pcts[second_group] + 1e4, axis = 0, 
                        weights = (non_missing_pcts[second_group] + 1e4)**scale_factor
                    )
                )
            
                second_group_weights.append(
                    np.average(
                        weights[second_group], axis=0,
                        weights=(ms_cell_weights[second_group] + 1e-4) ** scale_factor
                    )
                )
            
            second_cross_group_weights = np.array(second_cross_group_weights)
            second_group_weights = np.array(second_group_weights)
            
            # final weights for first level group
            second_final_weights = np.average(
                                        second_group_weights, axis=0,
                                        weights=second_cross_group_weights ** scale_factor
                                    )

            # first level aggregation -> cross group: missing_mech_sim
            first_cross_group_weights.append(
                np.average(
                    mech_sim_dist[client_idx][group] + 0.00001,axis= 0, 
                    weights= (mech_sim_dist[client_idx][group] + 0.00001)** scale_factor
                ) 
            )
           
            first_group_weights.append(second_final_weights)

        first_cross_group_weights = np.array(first_cross_group_weights)
        first_group_weights = np.array(first_group_weights)

        final_w = np.average(first_group_weights, axis=0, weights=first_cross_group_weights ** scale_factor)
        final_parameters.append(final_w)

    return final_parameters


def mech_cos_sim_distance(coef_client, coefs_centroid):
    weights = []
    for coef_centroid in coefs_centroid:
        coef_df1 = pd.Series(coef_centroid)
        coef_df2 = pd.Series(coef_client)
        weight = coef_df1.corr(
            coef_df2, method=lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
        )
        weights.append(weight)
    weights = np.array(weights)
    weights = 1 - (weights + 1) / 2
    return weights


def mech_cos_sim_matrix(ms_coefs):
    coef_df = pd.DataFrame([coef for coef in ms_coefs]).T
    weight = coef_df.corr(
        method=lambda x, y: np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 0.000001)
    ).values
    weight = 1 - (weight + 1) / 2
    return weight


def clustering(ms_coefs, threshold = 0.1):
    # sample_size -> (num_clients, num_features)
    coef_df = pd.DataFrame([coef for coef in ms_coefs])
    agg = AgglomerativeClustering(
        n_clusters=None, metric='cosine', linkage='average', distance_threshold=threshold
    )
    cluster_labels = agg.fit_predict(coef_df.values)
    # cluster groups
    groups = [[] for _ in range(len(set(cluster_labels)))]
    for i in range(coef_df.shape[0]):
        cluster_idx = cluster_labels[i]
        groups[cluster_idx].append(i)

    # centroid clusters
    centroid_groups = []
    for group in groups:
        centroid = np.mean(coef_df.iloc[group].values, axis=0)
        centroid_groups.append(centroid)

    return groups, centroid_groups


def clustering1(weights: np.ndarray, indices: List, threshold=0.1) -> List[List[int]]:
    if len(weights) == 1:
        return [indices]

    agg = AgglomerativeClustering(
        n_clusters=None, metric='l1', linkage='average', distance_threshold=threshold
    )
    cluster_labels = agg.fit_predict(weights.reshape(-1, 1))
    groups = [[] for _ in range(len(set(cluster_labels)))]
    for i in range(len(weights)):
        cluster_idx = cluster_labels[i]
        groups[cluster_idx].append(indices[i])

    return groups

def clustering2(weights: np.ndarray, threshold=0.1) -> List[List[int]]:
    if len(weights) == 1:
        return [[0]]

    agg = AgglomerativeClustering(
        n_clusters=None, metric='l1', linkage='average', distance_threshold=threshold
    )
    cluster_labels = agg.fit_predict(weights.reshape(-1, 1))
    groups = [[] for _ in range(len(set(cluster_labels)))]
    for i in range(len(weights)):
        cluster_idx = cluster_labels[i]
        groups[cluster_idx].append(i)

    return groups
