from sklearn.decomposition import PCA
import numpy as np
import ot
import pandas as pd
from typing import Union
from ot import sliced_wasserstein_distance
import seaborn as sns
import matplotlib.pyplot as plt


class DistanceComputation:
    
    @staticmethod
    def preprocess_dataset(data: Union[pd.DataFrame, np.ndarray], pca_col_threshold = 20):
        
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        # if the number of columns is greater than the threshold, perform PCA
        if data.shape[1] > pca_col_threshold:
            pca = PCA(n_components=pca_col_threshold)
            data = pca.fit_transform(data)
        
        return data
        
    @staticmethod
    def compute_distance(
        data1, data2, 
        distance_method = 'swd', 
        pca_col_threshold = 20
    ):
        
        assert data1.shape == data2.shape, "Data dimensions must match"
        
        if distance_method == 'swd':
            data1 = DistanceComputation.preprocess_dataset(data1, pca_col_threshold)
            data2 = DistanceComputation.preprocess_dataset(data2, pca_col_threshold)
            return sliced_wasserstein_distance(
                data1, data2,
                n_projections=300,
                seed=21
            )
        # elif distance_method == 'wasserstein':
        #     a = np.ones(len(data1)) / len(data1)
        #     b = np.ones(len(data2)) / len(data2)
        #     M = ot.dist(data1, data2)
        #     return ot.emd2(a, b, M)
        # elif distance_method == 'sinkhorn':
        #     a = np.ones(len(data1)) / len(data1)
        #     b = np.ones(len(data2)) / len(data2)
        #     M = ot.dist(data1, data2)
        #     return ot.sinkhorn2(a, b, M, 0.01)
        elif distance_method == 'correlation':
            corr_data1 = np.corrcoef(data1.T)
            corr_data2 = np.corrcoef(data2.T)
            return np.linalg.norm(corr_data1 - corr_data2)
        else:
            raise ValueError(f"Unsupported distance method: {distance_method}")

    @staticmethod
    def compute_distance_matrix(
        data_list, 
        distance_method='swd', 
        pca_col_threshold = 20
    ):
        
        distance_matrix = np.zeros((len(data_list), len(data_list)))
        
        for i in range(len(data_list)):
            for j in range(i, len(data_list)):
                dist = DistanceComputation.compute_distance(
                    data_list[i], data_list[j], 
                    distance_method, 
                    pca_col_threshold
                )
                distance_matrix[i][j] = dist
                distance_matrix[j][i] = dist
        
        return distance_matrix
    
    @staticmethod
    def show_distance_matrix(distance_matrix: np.ndarray, ax: plt.Axes, fontsize: int = 18):
        
        if len(distance_matrix) > 10:
            clients_names = [f"C{i+1}" for i in range(len(distance_matrix))]
        else:
            clients_names = [f"C{i+1}" for i in range(len(distance_matrix))]
        
        if distance_matrix.max() > 0.4:
            vmax = distance_matrix.max()
        else:
            vmax = 0.3
        
        ax = sns.heatmap(
            distance_matrix,
            annot=False,
            cmap="flare",
            xticklabels=clients_names,
            yticklabels=clients_names,
            ax = ax,
            vmin=0,
            vmax=vmax,
            square = True,
            cbar_kws={'format': '{x:.2f}', 'shrink': 0.8}
        )
        
        return ax
