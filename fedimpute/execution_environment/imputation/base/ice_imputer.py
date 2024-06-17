from typing import Tuple
import numpy as np


class ICEImputerMixin:

    @staticmethod
    def get_clip_thresholds(data_utils):

        min_values = np.zeros(data_utils['n_features'])
        max_values = np.zeros(data_utils['n_features'])
        for i in range(data_utils['n_features']):
            if i < data_utils['num_cols']:
                min_values[i] = data_utils['col_stats'][i]['min']
                max_values[i] = data_utils['col_stats'][i]['max']
            else:
                min_values[i] = 0
                max_values[i] = data_utils['col_stats'][i]['num_class']

        return min_values, max_values

    def set_clip_thresholds(self, min_values, max_values):
        self.min_values = min_values
        self.max_values = max_values

    @staticmethod
    def get_visit_indices(visit_sequence, missing_mask):
        frac_of_missing_values = missing_mask.mean(axis=0)
        missing_values_idx = np.flatnonzero(frac_of_missing_values)

        if visit_sequence == 'roman':
            ordered_idx = missing_values_idx
        elif visit_sequence == 'arabic':
            ordered_idx = missing_values_idx[::-1]
        elif visit_sequence == 'random':
            ordered_idx = missing_values_idx.copy()
            np.random.shuffle(ordered_idx)
        elif visit_sequence == 'ascending':
            n = len(frac_of_missing_values) - len(missing_values_idx)
            ordered_idx = np.argsort(frac_of_missing_values, kind="mergesort")[n:]
        elif visit_sequence == 'descending':
            n = len(frac_of_missing_values) - len(missing_values_idx)
            ordered_idx = np.argsort(frac_of_missing_values, kind="mergesort")[n:][::-1]
        else:
            raise ValueError("Invalid choice for visit order: %s" % visit_sequence)

        return ordered_idx
