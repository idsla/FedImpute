import numpy as np


class JMImputerMixin:

    def __init__(self):
        super().__init__()

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
