from typing import Union

import numpy as np
import pandas as pd

from .helper import one_hot_encoding, ordering_features


def prep_data(data: Union[np.ndarray, pd.DataFrame], numerical_cols, target_col):

    data = ordering_features(data, numerical_cols, target_col)
    data = one_hot_encoding(data, len(numerical_cols))

    return data
